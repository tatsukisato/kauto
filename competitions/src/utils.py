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


def crop_and_save_images(
    meta_df: pd.DataFrame, 
    input_dir: Path, 
    output_dir: Path,
    mode: str = 'train'
):
    """Crop and save images based on bbox
    
    Args:
        meta_df: DataFrame containing metadata (including bbox info)
        input_dir: Directory containing original images
        output_dir: Directory to save cropped images
        mode: 'train' or 'test'. If 'train', uses x,y,w,h columns to crop.
              If 'test', assumes images are already cropped (pass-through not needed normally, 
              but for consistency).
    """
    import cv2
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(meta_df)} images...")
    
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        # Construct image path
        # Assuming folder structure based on user description.
        # usually user provides specific paths. 
        # For this competition, let's assume rel_path is useful if available, 
        # or we construct it.
        # Based on typical structure: input_dir / row['image_name']
        
        # However, for 'train', we often have full images and annotations.
        # We need to know where the full images are.
        # Let's assume standard atmaCup structure or what's in the dataframe.
        
        # In this task, "test data is provided as cropped images".
        # Train data is "full image + bbox".
        
        if mode == 'train':
            # Construct full image path based on naming convention
            # {quarter}__{angle}__{session}__{frame}.jpg
            # session and frame are 2-digit zero-padded
            filename = f"{row['quarter']}__{row['angle']}__{int(row['session']):02d}__{int(row['frame']):02d}.jpg"
            img_path = input_dir / filename

            if not img_path.exists():
                # Try finding in 'images' subdirectory if not in root
                img_path_sub = input_dir / 'images' / filename
                if img_path_sub.exists():
                    img_path = img_path_sub
                else:
                    # Skip if not found
                    continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Crop
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            
            # Ensure valid bbox
            h_img, w_img = img.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w <= 0 or h <= 0:
                continue
                
            crop = img[y:y+h, x:x+w]
            
            # Save
            # Use index or some unique ID for filename
            save_name = f"{row.name}.jpg" 
            save_path = output_dir / save_name
            cv2.imwrite(str(save_path), crop)
            
    print(f"Finished saving cropped images to {output_dir}")


def visualize_crop_samples(
    meta_df: pd.DataFrame, 
    image_dir: Path, 
    num_samples: int = 5
):
    """Visualize random samples of cropped images
    
    Args:
        meta_df: DataFrame with metadata
        image_dir: Directory containing cropped images
        num_samples: Number of samples to visualize
    """
    import cv2
    import matplotlib.pyplot as plt
    
    samples = meta_df.sample(n=num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        # Assuming saved by index
        img_name = f"{row.name}.jpg"
        img_path = image_dir / img_name
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            label = row['label_id'] if 'label_id' in row else '?'
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
            
    plt.tight_layout()
    plt.show()
