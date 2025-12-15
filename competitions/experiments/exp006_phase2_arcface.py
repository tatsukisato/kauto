# === Added for exp006 improvement ===
# Exp006: Improvements Phase 2 Step 2 (ArcFace Minimal Intro)
# Base: Exp006 Phase 2 Step 1
# Improvements:
# 1. Enable use_arcface=True in AtmaCupModel
# 2. Verify that training runs and loss decreases

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/input")
ROOT_DIR = Path("/kaggle/working/kauto/competitions") if IS_KAGGLE else Path(__file__).resolve().parents[1]
DATA_DIR = Path("/kaggle/input/atmacup22") if IS_KAGGLE else Path(__file__).resolve().parents[1]

# Clone repository if on Kaggle
if IS_KAGGLE:
    if not (Path("/kaggle/working/kauto").exists()):
        os.system("git clone https://github.com/tatsukisato/kauto.git /kaggle/working/kauto")
    sys.path.append(str(ROOT_DIR))
else:
    sys.path.append(str(ROOT_DIR))

try:
    from src.utils import (
        setup_directories,
        save_results,
        print_experiment_info,
        crop_and_save_images,
    )
    from src.dataset import AtmaCup22Dataset, MixedImageDataset
    from src.models import AtmaCupModel
    from src.generate_background import generate_background_samples
except ImportError:
    print("Warning: Custom modules not found.")

def main():
    exp_name = "exp006_phase2_arcface"
    description = "Phase 2 Step 2: ArcFace Minimal Intro. use_arcface=True."
    
    DEBUG = not IS_KAGGLE
    EPOCHS = 2 if DEBUG else 12 
    
    # Setup Directories
    dirs = setup_directories(
        base_dir=str(Path("/kaggle/working") if IS_KAGGLE else ROOT_DIR), 
        data_dir=str(DATA_DIR)
    )
    print_experiment_info(exp_name, description)
    
    # 1. Load Data
    raw_dir = dirs['raw']
    dataset_handler = AtmaCup22Dataset(data_dir=str(raw_dir))
    train_meta, test_meta = dataset_handler.load_data()
    train_meta['original_index'] = train_meta.index

    if DEBUG:
        print("!!! DEBUG MODE: Using small subset !!!")
        unique_quarters = train_meta['quarter'].unique()
        if len(unique_quarters) >= 2:
            q1_df = train_meta[train_meta['quarter'] == unique_quarters[0]].head(100)
            q2_df = train_meta[train_meta['quarter'] == unique_quarters[1]].head(100)
            train_meta = pd.concat([q1_df, q2_df])
        else:
            train_meta = train_meta.iloc[:200]
        test_meta = test_meta.iloc[:50]
        
    # 2. Check/Generate Player Crops
    crops_dir = dirs['processed'] / 'crops_train'
    if not list(crops_dir.glob("*.jpg")):
         # Assuming generated in Step 1, simple check
         print("Generating player crops...")
         crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')
    
    # 3. Check/Generate Background Crops
    bg_crops_dir = dirs['processed'] / 'crops_bg'
    bg_csv_path = dirs['processed'] / 'train_meta_background.csv'
    
    if not bg_csv_path.exists():
         # Basic generation if missing (Step 1 usually does this)
         print("Generating Background Samples...")
         bg_df = generate_background_samples(train_meta, dirs['raw'], bg_crops_dir, 1, -1)
    else:
         print("Loading existing Background Samples...")
         bg_df = pd.read_csv(bg_csv_path)
        
    bg_df['original_index'] = -1
    full_train_df = pd.concat([train_meta, bg_df], axis=0, ignore_index=True)

    # 4. Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 5. Validation Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    exp_output_dir = dirs['output'] / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    groups = full_train_df['quarter'].to_numpy()
    X = full_train_df.index.to_numpy()
    y = full_train_df['label_id'].astype(str).to_numpy()
    
    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))
    
    print(f"\n{'='*20} Hold-out Validation (20%) {'='*20}")
    train_df_fold = full_train_df.iloc[train_idx]
    val_df_fold = full_train_df.iloc[val_idx]
    
    crop_dirs = {'train': crops_dir, 'bg': bg_crops_dir}
    train_dataset = MixedImageDataset(train_df_fold, crop_dirs, transform=train_transform, mode='train')
    val_dataset = MixedImageDataset(val_df_fold, crop_dirs, transform=val_transform, mode='validation')
    
    batch_size = 32 if DEBUG else 256
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # 6. Model (12 Classes) - ARCFACE ENABLED
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}, AMP: {use_amp}")
    
    model = AtmaCupModel(
        num_classes=12, 
        pretrained=True, 
        freeze_backbone=True,
        use_arcface=True,         # [MODIFIED] Enable ArcFace
        use_embedding_head=False
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler(enabled=use_amp)
    
    best_score = 0.0
    best_model_path = model_dir / f"{exp_name}_best.pth"
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                # Pass targets for ArcFace!
                outputs = model(images, targets=labels)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_dataset)
        
        # Val
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type="cuda", enabled=use_amp):
                    # No targets in eval -> returns scaled cosine
                    outputs = model(images)
                    # For Loss calculation in Val with ArcFace, we technically need margin? 
                    # Usually Val Loss for ArcFace is calculated same as Train (with targets) OR with simple Cosine?
                    # Let's pass targets to see validation loss with margin.
                    # Note: Our model.forward(targets=None) returns cosine*s.
                    # criterion(cosine*s, labels) is valid CE Loss on cosine similarities.
                    # If we want comparable "margin loss", we should pass targets.
                    # But for "metric" (accuracy), we use no targets (inference mode).
                    
                    # Option A: Val Loss = Inference Loss (Cosine Softmax)
                    loss = criterion(outputs, labels) 
                    
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        # Validation Metrics
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        
        macro_f1_all = f1_score(val_labels, val_preds, average='macro')
        
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  [Overall] F1: {macro_f1_all:.4f}")

        if macro_f1_all > best_score:
            best_score = macro_f1_all
            torch.save(model.state_dict(), best_model_path)
            
        scheduler.step(macro_f1_all)
        
    print(f"Best Val F1 (Overall): {best_score:.4f}")
    save_results({'val_score_overall': best_score}, str(exp_output_dir), exp_name)

if __name__ == "__main__":
    main()
