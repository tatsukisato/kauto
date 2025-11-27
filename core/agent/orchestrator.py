"""Main orchestrator for autonomous experiment loop with LLM integration."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from .base import BaseAgent
from .tools import AgentTools
from ..experiment.runner import ExperimentRunner
from ..experiment.tracker import ExperimentTracker


class ExperimentOrchestrator(BaseAgent):
    """Orchestrates autonomous experiment loops using LLM."""
    
    def __init__(
        self,
        competition_name: str,
        base_dir: Path,
        llm_provider: str = "gemini",
        model_name: Optional[str] = None
    ):
        super().__init__(competition_name, base_dir)
        
        # Initialize components
        self.tools = AgentTools(base_dir, competition_name)
        self.runner = ExperimentRunner(base_dir)
        self.tracker = ExperimentTracker(self.comp_dir)
        
        # LLM setup
        self.llm_provider = llm_provider
        self.model_name = model_name or self._get_default_model(llm_provider)
        self._setup_llm()
        
        # Load prompts
        self.system_prompt = self.load_prompt("agent_system")
        self.hypothesis_prompt = self.load_prompt("hypothesis_generation")
        self.analysis_prompt = self.load_prompt("result_analysis")
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model name for provider."""
        defaults = {
            "gemini": "gemini-2.0-flash-exp",
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022"
        }
        return defaults.get(provider, "gemini-2.0-flash-exp")
    
    def _setup_llm(self):
        """Setup LLM client based on provider."""
        if self.llm_provider == "gemini":
            try:
                import google.generativeai as genai
                import os
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.llm_client = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError("google-generativeai not installed. Run: uv add google-generativeai")
        
        elif self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
            except ImportError:
                raise ImportError("openai not installed. Run: uv add openai")
        
        elif self.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic()
            except ImportError:
                raise ImportError("anthropic not installed. Run: uv add anthropic")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call LLM with prompt."""
        if self.llm_provider == "gemini":
            response = self.llm_client.generate_content(
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            )
            return response.text
        
        elif self.llm_provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system_prompt if system_prompt else "",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def run(self, max_iterations: int = 10, auto_submit: bool = False):
        """
        Main autonomous experiment loop.
        
        Args:
            max_iterations: Maximum number of iterations
            auto_submit: Whether to automatically submit improved results
        """
        print(f"Starting autonomous experiment loop for {self.competition_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"Auto-submit: {auto_submit}")
        print("-" * 80)
        
        # Initialize
        self._initialize()
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*80}\n")
            
            try:
                # 1. Generate hypothesis
                hypothesis = self._generate_hypothesis()
                print(f"Hypothesis: {hypothesis['description']}")
                
                # 2. Implement experiment
                exp_name = f"exp{str(iteration + 2).zfill(3)}_{hypothesis['name']}"
                implementation = self._implement_experiment(exp_name, hypothesis)
                
                if not implementation['success']:
                    print(f"Implementation failed: {implementation['error']}")
                    continue
                
                # 3. Run experiment
                print(f"\nRunning experiment: {exp_name}")
                result = self._run_experiment(exp_name)
                
                if not result['success']:
                    print(f"Experiment failed: {result.get('error', 'Unknown error')}")
                    continue
                
                # 4. Analyze results
                analysis = self._analyze_results(exp_name, hypothesis, result)
                print(f"\nAnalysis: {analysis['summary']}")
                
                # 5. Update state
                self.state['iteration'] = iteration + 1
                self.state['experiments_run'].append(exp_name)
                
                # Check for improvement
                if analysis.get('improved', False):
                    self.state['best_score'] = analysis['score']
                    print(f"\n✅ Improvement! New best score: {analysis['score']}")
                    
                    if auto_submit:
                        self._submit_results(exp_name, analysis)
                else:
                    print(f"\n❌ No improvement. Best score remains: {self.state.get('best_score', 'N/A')}")
                
                # 6. Save state
                self._save_state()
                
                # 7. Report
                self._report_iteration(iteration + 1, exp_name, hypothesis, analysis)
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print("Experiment loop completed")
        print(f"Total experiments run: {len(self.state['experiments_run'])}")
        print(f"Best score: {self.state.get('best_score', 'N/A')}")
        print(f"{'='*80}")
    
    def _initialize(self):
        """Initialize the agent by reading competition docs."""
        print("Initializing agent...")
        
        # Read competition docs
        readme = self.tools.read_file("docs/README.md")
        overview = self.tools.read_file("docs/OVERVIEW.md")
        
        # Get experiment history
        recent_exps = self.tracker.get_recent_experiments(n=5)
        
        self.state['competition_docs'] = {
            'readme': readme[:1000] if readme else "",  # Truncate for context
            'overview': overview[:1000] if overview else ""
        }
        self.state['recent_experiments'] = recent_exps
        
        print("Initialization complete")
    
    def _generate_hypothesis(self) -> Dict[str, Any]:
        """Generate experiment hypothesis using LLM."""
        print("\nGenerating hypothesis...")
        
        # Prepare context
        context = self.hypothesis_prompt.format(
            competition_name=self.competition_name,
            task_type="Classification",  # TODO: Extract from docs
            metric="Accuracy",  # TODO: Extract from docs
            data_description=self.state['competition_docs'].get('readme', ''),
            experiment_history=json.dumps(self.state.get('recent_experiments', []), indent=2),
            best_score=self.state.get('best_score', 'No experiments yet')
        )
        
        # Call LLM
        response = self._call_llm(context, self.system_prompt)
        
        # Parse response (simple parsing, can be improved)
        hypothesis = {
            'description': response[:500],  # Truncate
            'name': f"iteration_{self.state['iteration'] + 1}",
            'full_response': response
        }
        
        return hypothesis
    
    def _implement_experiment(self, exp_name: str, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Implement experiment based on hypothesis."""
        print(f"\nImplementing experiment: {exp_name}")
        
        # For now, use the existing baseline as template
        # In a full implementation, LLM would generate the code
        baseline_code = self.tools.read_file("experiments/exp001_baseline.py")
        
        if not baseline_code:
            return {"success": False, "error": "Could not read baseline code"}
        
        # Simple implementation: copy baseline with modified name
        # TODO: Have LLM generate actual modifications
        modified_code = baseline_code.replace("exp001_baseline", exp_name)
        
        # Write experiment script
        success = self.tools.write_code(f"experiments/{exp_name}.py", modified_code)
        
        return {"success": success}
    
    def _run_experiment(self, exp_name: str) -> Dict[str, Any]:
        """Run the experiment."""
        result = self.runner.run(
            competition_name=self.competition_name,
            exp_script=f"experiments/{exp_name}.py"
        )
        return result
    
    def _analyze_results(self, exp_name: str, hypothesis: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results using LLM."""
        print("\nAnalyzing results...")
        
        # Get experiment output
        exp_output = result.get('stdout', '')[-2000:]  # Last 2000 chars
        
        # Extract CV score (simple regex, can be improved)
        import re
        cv_match = re.search(r'Mean Accuracy: ([\d.]+)', exp_output)
        cv_score = float(cv_match.group(1)) if cv_match else None
        
        # Prepare context
        context = self.analysis_prompt.format(
            exp_name=exp_name,
            hypothesis=hypothesis['description'],
            implementation="Modified baseline",  # TODO: Actual implementation details
            experiment_output=exp_output,
            cv_score=cv_score if cv_score else "Not found",
            previous_best=self.state.get('best_score', 'N/A')
        )
        
        # Call LLM
        response = self._call_llm(context, self.system_prompt)
        
        # Determine if improved
        improved = False
        if cv_score and self.state.get('best_score'):
            improved = cv_score > self.state['best_score']
        elif cv_score and not self.state.get('best_score'):
            improved = True
        
        analysis = {
            'summary': response[:500],
            'score': cv_score,
            'improved': improved,
            'full_analysis': response
        }
        
        # Log to tracker
        self.tracker.log_experiment(
            exp_name=exp_name,
            config={'hypothesis': hypothesis['description']},
            metrics={'cv_score': cv_score} if cv_score else {},
            success=result['success'],
            notes=analysis['summary']
        )
        
        return analysis
    
    def _submit_results(self, exp_name: str, analysis: Dict[str, Any]):
        """Submit results to Kaggle."""
        print(f"\nSubmitting {exp_name} to Kaggle...")
        
        submission_path = f"submissions/{exp_name}_submission.csv"
        message = f"Auto-submit: {analysis['summary'][:100]}"
        
        result = self.tools.submit_to_kaggle(submission_path, message)
        
        if result['success']:
            print("✅ Submission successful")
        else:
            print(f"❌ Submission failed: {result.get('error', 'Unknown error')}")
    
    def _report_iteration(self, iteration: int, exp_name: str, hypothesis: Dict[str, Any], analysis: Dict[str, Any]):
        """Report iteration results."""
        report = f"""
## イテレーション {iteration} 報告

### 実行した実験
- 実験名: {exp_name}
- 仮説: {hypothesis['description'][:200]}

### 結果
- CVスコア: {analysis.get('score', 'N/A')}
- 改善: {'はい' if analysis.get('improved') else 'いいえ'}

### 分析
{analysis['summary'][:300]}
"""
        print(report)
        
        # Save report
        report_file = self.comp_dir / "output" / f"iteration_{iteration}_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report, encoding='utf-8')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run autonomous experiment orchestrator")
    parser.add_argument("--competition", required=True, help="Competition name")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--auto-submit", action="store_true", help="Auto-submit improvements")
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--model", help="Model name (optional)")
    
    args = parser.parse_args()
    
    orchestrator = ExperimentOrchestrator(
        competition_name=args.competition,
        base_dir=Path.cwd(),
        llm_provider=args.llm_provider,
        model_name=args.model
    )
    
    orchestrator.run(
        max_iterations=args.max_iterations,
        auto_submit=args.auto_submit
    )
