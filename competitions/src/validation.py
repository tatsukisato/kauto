"""
Validation module for atmaCup #22
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from typing import Tuple


def create_time_split(
    train_meta: pd.DataFrame,
    val_quarters: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time-based train/validation split
    
    Args:
        train_meta: Training metadata DataFrame
        val_quarters: List of quarter numbers to use for validation (e.g., [4])
        
    Returns:
        train_df: Training split
        val_df: Validation split
    """
    if val_quarters is None:
        val_quarters = [4]  # Use Q4 for validation by default
    
    # Extract quarter number from quarter column (e.g., "Q1-000" -> 1)
    quarter_nums = train_meta['quarter'].str.extract(r'Q(\d+)')[0].astype(int)
    
    # Split based on quarter
    val_mask = quarter_nums.isin(val_quarters)
    train_df = train_meta[~val_mask].copy()
    val_df = train_meta[val_mask].copy()
    
    print(f"Train split: {len(train_df)} samples")
    print(f"Validation split: {len(val_df)} samples")
    print(f"Train quarters: {sorted(quarter_nums[~val_mask].unique())}")
    print(f"Val quarters: {sorted(quarter_nums[val_mask].unique())}")
    
    return train_df, val_df


def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Macro F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro')


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict = None
) -> dict:
    """Evaluate predictions and return detailed metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional mapping of label IDs to names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate macro F1
    macro_f1 = calculate_macro_f1(y_true, y_pred)
    
    # Calculate per-class F1
    unique_labels = sorted(set(y_true) | set(y_pred))
    per_class_f1 = f1_score(y_true, y_pred, labels=unique_labels, average=None)
    
    results = {
        'macro_f1': macro_f1,
        'per_class_f1': {}
    }
    
    print(f"\n{'='*50}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"{'='*50}")
    print("\nPer-class F1 scores:")
    print(f"{'Label':<10} {'F1 Score':<10} {'Support':<10}")
    print(f"{'-'*30}")
    
    for label, f1 in zip(unique_labels, per_class_f1):
        support = (y_true == label).sum()
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        results['per_class_f1'][int(label)] = float(f1)  # Convert to native Python types
        print(f"{label_name:<10} {f1:<10.4f} {support:<10}")
    
    return results
