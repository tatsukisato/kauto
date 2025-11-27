"""Base agent class for autonomous Kaggle experimentation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import json


class BaseAgent(ABC):
    """Base class for autonomous agents."""
    
    def __init__(self, competition_name: str, base_dir: Path):
        self.competition_name = competition_name
        self.base_dir = base_dir
        self.comp_dir = base_dir / "competitions" / competition_name
        self.state_dir = base_dir / "core" / "agent" / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load agent state from disk."""
        state_file = self.state_dir / f"{self.competition_name}_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        return {
            "iteration": 0,
            "experiments_run": [],
            "best_score": None,
            "hypotheses": []
        }
    
    def _save_state(self):
        """Save agent state to disk."""
        state_file = self.state_dir / f"{self.competition_name}_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        prompt_file = self.base_dir / "prompts" / f"{prompt_name}.md"
        if prompt_file.exists():
            return prompt_file.read_text(encoding='utf-8')
        return ""
    
    @abstractmethod
    def run(self, max_iterations: int = 10):
        """Main agent loop."""
        pass
