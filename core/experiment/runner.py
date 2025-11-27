"""Experiment runner with configuration management."""

from pathlib import Path
from typing import Dict, Any
import subprocess
import json
from datetime import datetime


class ExperimentRunner:
    """Generic experiment runner with Hydra config support."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    
    def run(self, competition_name: str, exp_script: str, config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run an experiment with optional config overrides.
        
        Args:
            competition_name: Name of the competition
            exp_script: Path to experiment script relative to competition dir
            config_overrides: Optional dict of config overrides
        
        Returns:
            Dict containing execution results
        """
        comp_dir = self.base_dir / "competitions" / competition_name
        script_path = comp_dir / exp_script
        
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script not found: {exp_script}"
            }
        
        # Build command
        cmd = ["uv", "run", "python", str(script_path)]
        
        # Add Hydra overrides if provided
        if config_overrides:
            for key, value in config_overrides.items():
                cmd.append(f"{key}={value}")
        
        # Execute
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Experiment timed out after 1 hour"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
