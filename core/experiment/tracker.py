"""Experiment tracking and history management."""

from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


class ExperimentTracker:
    """Track experiments and maintain history."""
    
    def __init__(self, competition_dir: Path):
        self.competition_dir = competition_dir
        self.history_file = competition_dir / "output" / "experiment_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load experiment history from disk."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save experiment history to disk."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_experiment(
        self,
        exp_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        success: bool,
        notes: str = ""
    ):
        """
        Log an experiment to history.
        
        Args:
            exp_name: Experiment name
            config: Experiment configuration
            metrics: Experiment metrics (e.g., CV score)
            success: Whether experiment succeeded
            notes: Optional notes
        """
        entry = {
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "success": success,
            "notes": notes
        }
        
        self.history.append(entry)
        self._save_history()
    
    def get_best_experiment(self, metric_name: str = "cv_score") -> Dict[str, Any]:
        """
        Get the best experiment based on a metric.
        
        Args:
            metric_name: Name of metric to compare
        
        Returns:
            Best experiment entry
        """
        successful_exps = [e for e in self.history if e["success"]]
        if not successful_exps:
            return None
        
        return max(
            successful_exps,
            key=lambda e: e["metrics"].get(metric_name, float('-inf'))
        )
    
    def get_recent_experiments(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent experiments."""
        return self.history[-n:] if len(self.history) >= n else self.history
    
    def compare_experiments(self, exp_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            exp_names: List of experiment names to compare
        
        Returns:
            Comparison results
        """
        experiments = [e for e in self.history if e["exp_name"] in exp_names]
        
        if not experiments:
            return {"error": "No experiments found"}
        
        comparison = {
            "experiments": experiments,
            "best": max(experiments, key=lambda e: e["metrics"].get("cv_score", float('-inf')))
        }
        
        return comparison
