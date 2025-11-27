from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class BaseFeature(ABC):
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Execute feature engineering logic.
        Should return a DataFrame containing the new features.
        """
        pass

    def save(self, df: pd.DataFrame, filename: str):
        """Save the feature dataframe."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / filename, index=False)
        print(f"Saved {self.name} features to {self.output_dir / filename}")
