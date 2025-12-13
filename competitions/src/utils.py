"""
Utility functions for atmaCup #22
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def setup_directories(base_dir: str = ".") -> dict:
    """Setup directory structure
    
    Args:
        base_dir: Base directory
        
    Returns:
        Dictionary with directory paths
    """
    base_path = Path(base_dir)
    
    dirs = {
        'data': base_path / 'data',
        'raw': base_path / 'data' / 'raw',
        'processed': base_path / 'data' / 'processed',
        'output': base_path / 'output',
        'models': base_path / 'output' / 'models',
        'submissions': base_path / 'submissions',
    }
    
    # Create directories if they don't exist
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_results(results: dict, output_dir: str, exp_name: str):
    """Save experiment results
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        exp_name: Experiment name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    results['experiment'] = exp_name
    
    # Save as JSON
    json_path = output_path / f"{exp_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {json_path}")


def create_submission(
    predictions: list,
    output_path: str,
    test_meta: pd.DataFrame = None
):
    """Create submission file
    
    Args:
        predictions: List of predictions
        output_path: Path to save submission file
        test_meta: Test metadata (optional, for validation)
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'label_id': predictions
    })
    
    # Validate
    if test_meta is not None:
        assert len(submission) == len(test_meta), \
            f"Submission length {len(submission)} != test length {len(test_meta)}"
    
    # Check value range
    assert submission['label_id'].min() >= -1, "Invalid label: < -1"
    assert submission['label_id'].max() <= 10, "Invalid label: > 10"
    
    # Save without header and index (only label_id column)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, header=True)
    
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Label distribution:\n{submission['label_id'].value_counts().sort_index()}")


def print_experiment_info(exp_name: str, description: str = ""):
    """Print experiment information
    
    Args:
        exp_name: Experiment name
        description: Experiment description
    """
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    if description:
        print(f"Description: {description}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
