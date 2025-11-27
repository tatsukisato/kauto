"""Tool definitions for the autonomous agent."""

from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import json


class AgentTools:
    """Collection of tools available to the agent."""
    
    def __init__(self, base_dir: Path, competition_name: str):
        self.base_dir = base_dir
        self.competition_name = competition_name
        self.comp_dir = base_dir / "competitions" / competition_name
    
    def run_experiment(self, exp_script: str) -> Dict[str, Any]:
        """
        Execute an experiment script.
        
        Args:
            exp_script: Path to experiment script relative to competition dir
        
        Returns:
            Dict containing execution results and metrics
        """
        script_path = self.comp_dir / exp_script
        if not script_path.exists():
            return {"success": False, "error": f"Script not found: {exp_script}"}
        
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script_path)],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Experiment timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read a file from the competition directory.
        
        Args:
            file_path: Path relative to competition directory
        
        Returns:
            File contents or None if not found
        """
        full_path = self.comp_dir / file_path
        if full_path.exists():
            return full_path.read_text(encoding='utf-8')
        return None
    
    def write_code(self, file_path: str, content: str) -> bool:
        """
        Write code to a file in the competition directory.
        
        Args:
            file_path: Path relative to competition directory
            content: Code content to write
        
        Returns:
            True if successful, False otherwise
        """
        try:
            full_path = self.comp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False
    
    def analyze_results(self, exp_name: str) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            exp_name: Experiment name (e.g., 'exp001_baseline')
        
        Returns:
            Dict containing analysis results
        """
        output_dir = self.comp_dir / "output" / exp_name
        if not output_dir.exists():
            return {"success": False, "error": f"Output directory not found: {exp_name}"}
        
        # Look for metrics file or parse logs
        metrics = {}
        
        # Try to find submission file
        submission_dir = self.comp_dir / "submissions"
        submission_file = submission_dir / f"{exp_name}_submission.csv"
        
        return {
            "success": True,
            "output_dir": str(output_dir),
            "submission_exists": submission_file.exists(),
            "metrics": metrics
        }
    
    def submit_to_kaggle(self, submission_path: str, message: str) -> Dict[str, Any]:
        """
        Submit predictions to Kaggle.
        
        Args:
            submission_path: Path to submission file relative to competition dir
            message: Submission message
        
        Returns:
            Dict containing submission results
        """
        full_path = self.comp_dir / submission_path
        if not full_path.exists():
            return {"success": False, "error": f"Submission file not found: {submission_path}"}
        
        try:
            result = subprocess.run(
                [
                    "uv", "run", "python",
                    str(self.base_dir / "core" / "utils" / "submitter.py"),
                    self.competition_name,
                    str(full_path),
                    message
                ],
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
