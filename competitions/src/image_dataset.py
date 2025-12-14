
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    """Dataset for atmaCup #22 image data"""
    
    def __init__(self, meta_df: pd.DataFrame, image_dir: str, transform=None, mode='train'):
        """
        Args:
            meta_df: DataFrame with metadata
            image_dir: Directory containing images
                       For train/val, this should be the directory of CROPPED images.
                       For test, this should be the root data directory (containing 'crops/...' paths).
            transform: Albumentations or Torchvision transforms
            mode: 'train', 'validation', or 'test'
        """
        self.meta_df = meta_df
        # image_dir should be the directory with CROPPED images
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        
        # Determine image path
        if self.mode == 'test' and 'rel_path' in row:
             # Test data already has relative path in 'rel_path' column
             # The image_dir should be 'data/raw' or wherever the base for rel_path is.
             img_path = self.image_dir / row['rel_path']
        else:
             # Train/Val data (our generated crops)
             # We assume they are saved as f"{row.name}.jpg" in image_dir
             # Be careful: row.name is the index. Ensure metadata used here preserves original index 
             # corresponding to how images were saved.
             img_path = self.image_dir / f"{row.name}.jpg"
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
             # Fallback: return black image
             print(f"Warning: Image not found {img_path}")
             img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        # Check if transform is a torchvision transform (callable)
        if self.transform:
            img = self.transform(img)
            
        if self.mode == 'train' or self.mode == 'validation':
            label = int(row['label_id'])
            # Ensure label is valid (0-10)
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img
