"""MCP server for kauto - Kaggle autonomous agent tools."""

import json
from pathlib import Path
from typing import Any
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from core.agent.tools import AgentTools
from core.experiment.runner import ExperimentRunner
from core.experiment.tracker import ExperimentTracker


class KaggleAgentMCP:
    """MCP server for Kaggle agent tools."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.current_competition = None
        self.tools = None
        self.runner = None
        self.tracker = None
    
    def set_competition(self, competition_name: str) -> dict[str, Any]:
        """Set the current competition context."""
        self.current_competition = competition_name
        self.tools = AgentTools(self.base_dir, competition_name)
        self.runner = ExperimentRunner(self.base_dir)
        comp_dir = self.base_dir / "competitions" / competition_name
        self.tracker = ExperimentTracker(comp_dir)
        
        return {
            "success": True,
            "competition": competition_name,
            "message": f"Competition set to {competition_name}"
        }
    
    def run_experiment(self, exp_script: str) -> dict[str, Any]:
        """Run an experiment script."""
        if not self.tools:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        return self.tools.run_experiment(exp_script)
    
    def read_file(self, file_path: str, competition_name: str = None) -> dict[str, Any]:
        """Read a file from the competition directory."""
        # Allow passing competition_name for stateless calls
        if competition_name:
            self.set_competition(competition_name)
        
        if not self.tools:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        content = self.tools.read_file(file_path)
        if content is None:
            return {"success": False, "error": f"File not found: {file_path}"}
        
        return {"success": True, "content": content}
    
    def write_code(self, file_path: str, content: str) -> dict[str, Any]:
        """Write code to a file."""
        if not self.tools:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        success = self.tools.write_code(file_path, content)
        return {
            "success": success,
            "message": f"File written to {file_path}" if success else "Failed to write file"
        }
    
    def analyze_results(self, exp_name: str) -> dict[str, Any]:
        """Analyze experiment results."""
        if not self.tools:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        return self.tools.analyze_results(exp_name)
    
    def submit_to_kaggle(self, submission_path: str, message: str) -> dict[str, Any]:
        """Submit to Kaggle."""
        if not self.tools:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        return self.tools.submit_to_kaggle(submission_path, message)
    
    def get_experiment_history(self, n: int = 5, competition_name: str = None) -> dict[str, Any]:
        """Get recent experiment history."""
        if competition_name:
            self.set_competition(competition_name)
        
        if not self.tracker:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        recent = self.tracker.get_recent_experiments(n)
        return {"success": True, "experiments": recent}
    
    def get_best_experiment(self, metric_name: str = "cv_score", competition_name: str = None) -> dict[str, Any]:
        """Get the best experiment."""
        if competition_name:
            self.set_competition(competition_name)
        
        if not self.tracker:
            return {"success": False, "error": "No competition set. Call set_competition first."}
        
        best = self.tracker.get_best_experiment(metric_name)
        if best is None:
            return {"success": False, "message": "No successful experiments found"}
        
        return {"success": True, "best_experiment": best}
    
    def list_available_tools(self) -> dict[str, Any]:
        """List all available tools."""
        tools = [
            {
                "name": "set_competition",
                "description": "Set the current competition context",
                "parameters": {"competition_name": "string"}
            },
            {
                "name": "run_experiment",
                "description": "Run an experiment script",
                "parameters": {"exp_script": "string (path relative to competition dir)"}
            },
            {
                "name": "read_file",
                "description": "Read a file from the competition directory",
                "parameters": {"file_path": "string"}
            },
            {
                "name": "write_code",
                "description": "Write code to a file",
                "parameters": {"file_path": "string", "content": "string"}
            },
            {
                "name": "analyze_results",
                "description": "Analyze experiment results",
                "parameters": {"exp_name": "string"}
            },
            {
                "name": "submit_to_kaggle",
                "description": "Submit predictions to Kaggle",
                "parameters": {"submission_path": "string", "message": "string"}
            },
            {
                "name": "get_experiment_history",
                "description": "Get recent experiment history",
                "parameters": {"n": "int (default: 5)"}
            },
            {
                "name": "get_best_experiment",
                "description": "Get the best experiment by metric",
                "parameters": {"metric_name": "string (default: cv_score)"}
            }
        ]
        
        return {"success": True, "tools": tools}


# MCP server instance
mcp_server = KaggleAgentMCP()


def handle_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Handle MCP requests."""
    try:
        if method == "set_competition":
            return mcp_server.set_competition(**params)
        elif method == "run_experiment":
            return mcp_server.run_experiment(**params)
        elif method == "read_file":
            return mcp_server.read_file(**params)
        elif method == "write_code":
            return mcp_server.write_code(**params)
        elif method == "analyze_results":
            return mcp_server.analyze_results(**params)
        elif method == "submit_to_kaggle":
            return mcp_server.submit_to_kaggle(**params)
        elif method == "get_experiment_history":
            return mcp_server.get_experiment_history(**params)
        elif method == "get_best_experiment":
            return mcp_server.get_best_experiment(**params)
        elif method == "list_tools":
            return mcp_server.list_available_tools()
        else:
            return {"success": False, "error": f"Unknown method: {method}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle Agent MCP Server")
    parser.add_argument("--method", required=True, help="Method to call")
    parser.add_argument("--params", help="JSON params")
    
    args = parser.parse_args()
    
    params = json.loads(args.params) if args.params else {}
    result = handle_request(args.method, params)
    
    print(json.dumps(result, indent=2))
