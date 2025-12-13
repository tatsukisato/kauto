"""
Dataset module for atmaCup #22
"""
import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np


class AtmaCup22Dataset:
    """Dataset class for basketball player identification"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.train_meta = None
        self.test_meta = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test metadata"""
        self.train_meta = pd.read_csv(self.data_dir / "train_meta.csv")
        self.test_meta = pd.read_csv(self.data_dir / "test_meta.csv")
        
        print(f"Train data shape: {self.train_meta.shape}")
        print(f"Test data shape: {self.test_meta.shape}")
        
        return self.train_meta, self.test_meta
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from metadata
        
        Args:
            df: DataFrame with metadata (train or test)
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame()
        
        # Bbox position features
        features['x'] = df['x']
        features['y'] = df['y']
        features['w'] = df['w']
        features['h'] = df['h']
        
        # Bbox derived features
        features['area'] = df['w'] * df['h']
        features['aspect_ratio'] = df['w'] / (df['h'] + 1e-6)
        features['center_x'] = df['x'] + df['w'] / 2
        features['center_y'] = df['y'] + df['h'] / 2
        
        # Quarter encoding (extract number from Q1-000 format)
        features['quarter_num'] = df['quarter'].str.extract(r'Q(\d+)')[0].astype(int)
        features['quarter_sub'] = df['quarter'].str.extract(r'-(\d+)')[0].astype(int)
        
        # Session and frame
        features['session'] = df['session']
        features['frame'] = df['frame']
        
        # Angle encoding (side=0, top=1)
        features['angle'] = (df['angle'] == 'top').astype(int)
        
        return features
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training features and labels
        
        Returns:
            X: Features
            y: Labels
        """
        if self.train_meta is None:
            self.load_data()
            
        X = self.create_features(self.train_meta)
        y = self.train_meta['label_id']
        
        return X, y
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test features
        
        Returns:
            X: Features
        """
        if self.test_meta is None:
            self.load_data()
            
        X = self.create_features(self.test_meta)
        
        return X
