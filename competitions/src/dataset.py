"""
Dataset module for atmaCup #22
"""
import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


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

class MixedImageDataset(Dataset):
    """Dataset combining player crops and background samples.

    Expects meta_df to contain 'label_id' and optionally 'original_index' or 'file_name'
    for background rows. Background label_id == -1 is mapped to class 11.
    """
    def __init__(self, meta_df: pd.DataFrame, crop_dirs: dict, transform=None, mode: str = "train"):
        self.meta_df = meta_df.reset_index(drop=True)
        self.crop_dirs = crop_dirs
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        label = int(row["label_id"])

        if label == -1:
            fname = row.get("file_name", f"bg_{row.name}.jpg")
            if pd.isna(fname):
                fname = f"bg_{row.name}.jpg"
            img_path = Path(self.crop_dirs["bg"]) / fname
        else:
            idx_name = row.get("original_index", row.name)
            img_path = Path(self.crop_dirs["train"]) / f"{idx_name}.jpg"

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.mode in ("train", "validation"):
            target = 11 if label == -1 else label
            return img, torch.tensor(target, dtype=torch.long)
        return img
