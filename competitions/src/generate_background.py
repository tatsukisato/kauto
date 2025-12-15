
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import setup_directories

def iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 1e-6:
        return 0
    return inter_area / (union_area + 1e-6)

def is_overlapping(new_box, existing_boxes, threshold=0.1):
    """Check if new_box overlaps significantly with any existing_boxes"""
    for box in existing_boxes:
        if iou(new_box, box) > threshold:
            return True
    return False

def generate_background_samples(
    train_meta: pd.DataFrame, 
    raw_dir: Path, 
    output_dir: Path, 
    samples_per_image: int = 1,
    bg_label: int = -1
):
    """Generate background samples from training images"""
    import random
    random.seed(42)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by image file (unique frame)
    # Key: (quarter, angle, session, frame)
    # But wait, filename construction logic is in utils. Let's reuse or reimplement simple one.
    # We need to group by the logical image key.
    
    # Create image identifier
    train_meta['img_key'] = train_meta.apply(
        lambda r: f"{r['quarter']}__{r['angle']}__{int(r['session']):02d}__{int(r['frame']):02d}", axis=1
    )
    
    unique_images = train_meta['img_key'].unique()
    print(f"Found {len(unique_images)} unique images.")
    
    new_rows = []
    
    for img_key in tqdm(unique_images, desc="Generating Background"):
        # Get all existing bboxes for this image
        img_df = train_meta[train_meta['img_key'] == img_key]
        
        # Construct filename
        filename = f"{img_key}.jpg"
        img_path = raw_dir / filename
        if not img_path.exists():
            img_path = raw_dir / 'images' / filename
            if not img_path.exists():
                continue
                
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img, _ = img.shape
        
        existing_boxes = img_df[['x', 'y', 'w', 'h']].values.tolist()
        
        # Average size of player bboxes to generate similar sized background crops
        avg_w = int(img_df['w'].mean())
        avg_h = int(img_df['h'].mean())
        
        generated_count = 0
        attempts = 0
        max_attempts = 100 # Increased attempts for harder constraints
        
        while generated_count < samples_per_image and attempts < max_attempts:
            attempts += 1
            
            # Strategy: Mix of Pure Background and Shifted Player Crops
            # Test data contains:
            # 1. Non-player objects (Pure Background)
            # 2. Poorly detected players (IoU < 0.5) -> Should be -1
            
            is_shifted = (random.random() < 0.5) # 50% chance to try generating a shifted box
            
            if is_shifted and len(existing_boxes) > 0:
                # Generate a "Bad" BBox near a real player
                target_box = random.choice(existing_boxes)
                tx, ty, tw, th = target_box
                
                # Shift and Resize randomly
                # Shift by 30-70% of width/height to get low IoU
                shift_x = int(tw * random.uniform(-0.8, 0.8))
                shift_y = int(th * random.uniform(-0.8, 0.8))
                
                scale_w = random.uniform(0.8, 1.2)
                scale_h = random.uniform(0.8, 1.2)
                
                w_crop = int(tw * scale_w)
                h_crop = int(th * scale_h)
                x_crop = tx + shift_x
                y_crop = ty + shift_y
                
            else:
                # Pure Random Crop (Background)
                w_crop = int(avg_w * random.uniform(0.8, 1.2))
                h_crop = int(avg_h * random.uniform(0.8, 1.2))
                x_crop = random.randint(0, max(1, w_img - w_crop))
                y_crop = random.randint(0, max(1, h_img - h_crop))

            # Bound checks
            if w_crop <= 0 or h_crop <= 0: continue
            x_crop = max(0, min(x_crop, w_img - w_crop))
            y_crop = max(0, min(y_crop, h_img - h_crop))
            w_crop = min(w_crop, w_img - x_crop)
            h_crop = min(h_crop, h_img - y_crop)
            
            new_box = [x_crop, y_crop, w_crop, h_crop]
            
            # Check overlap with ALL players
            # Definition of Label -1: IoU < 0.5 with ANY valid player
            # If IoU >= 0.5 with any player, it count as that player (Label 0-10) -> Skip
            
            max_iou = 0.0
            for box in existing_boxes:
                iou_val = iou(new_box, box)
                max_iou = max(max_iou, iou_val)
                
            if max_iou < 0.5:
                # This is a valid Negative Sample (Label -1)
                # It could be pure background (IoU=0) or partial player (IoU < 0.5)
                
                crop = img[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                
                # Verify crop is valid image
                if crop.size == 0: continue
                
                save_name = f"bg_{img_key}_{generated_count}.jpg"
                cv2.imwrite(str(output_dir / save_name), crop)
                
                # Metadata
                base_row = img_df.iloc[0].copy()
                base_row['label_id'] = bg_label
                base_row['x'] = x_crop
                base_row['y'] = y_crop
                base_row['w'] = w_crop
                base_row['h'] = h_crop
                base_row['file_name'] = save_name
                
                # Add info about generation type for analysis if needed
                base_row['gen_type'] = 'shifted' if (is_shifted and max_iou > 0) else 'random'
                base_row['max_iou'] = max_iou
                
                new_rows.append(base_row)
                generated_count += 1
                
    bg_df = pd.DataFrame(new_rows)
    print(f"Generated {len(bg_df)} background samples.")
    
    # Save metadata csv
    bg_df.to_csv(output_dir.parent / "train_meta_background.csv", index=False)
    return bg_df

if __name__ == "__main__":
    dirs = setup_directories()
    
    # Load original train meta
    meta_path = dirs['raw'] / 'train_meta.csv'
    if not meta_path.exists():
        print("train_meta.csv not found.")
        sys.exit(1)
        
    train_meta = pd.read_csv(meta_path)
    
    # Output directory for BG crops
    bg_crops_dir = dirs['processed'] / 'crops_bg'
    
    generate_background_samples(
        train_meta=train_meta,
        raw_dir=dirs['raw'],
        output_dir=bg_crops_dir,
        samples_per_image=1, # 1 BG per image ~ 4000 samples (20% of 25k) roughly
        bg_label=-1
    )
