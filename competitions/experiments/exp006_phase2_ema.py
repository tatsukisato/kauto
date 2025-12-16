# === Added for exp006 improvement ===
# Exp006: Improvements Phase 2 Step 4 (EMA Implementation)
# Base: Exp006 Phase 2 Step 1
# Improvements:
# 1. Integrate ModelEMA
# 2. Log both Normal and EMA validation scores

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import cv2

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/input")
ROOT_DIR = Path("/kaggle/working/kauto/competitions") if IS_KAGGLE else Path(__file__).resolve().parents[1]
DATA_DIR = Path("/kaggle/input/atmacup22") if IS_KAGGLE else Path(__file__).resolve().parents[1]

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
        create_submission,
        print_experiment_info,
        crop_and_save_images,
    )
    from src.dataset import AtmaCup22Dataset, MixedImageDataset
    from src.models import AtmaCupModel
    from src.generate_background import generate_background_samples
    from src.utils_ema import ModelEMA  # [MODIFIED] Import EMA
    from src.metrics import compute_evaluation_metrics
except ImportError:
    print("Warning: Custom modules not found.")

def validate(model, val_loader, device, use_amp, criterion, desc="Val"):
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    metrics = compute_evaluation_metrics(val_labels, val_preds, bg_label=11)
    return val_loss, metrics

def main():
    exp_name = "exp006_phase2_ema"
    description = "Phase 2 Step 4: EMA Implementation. Verify EMA score vs Normal score."
    
    DEBUG = not IS_KAGGLE
    EPOCHS = 2 if DEBUG else 12 
    
    dirs = setup_directories(base_dir=str(Path("/kaggle/working") if IS_KAGGLE else ROOT_DIR), data_dir=str(DATA_DIR))
    print_experiment_info(exp_name, description)
    
    # Load Data (Simplified for brevity)
    dataset_handler = AtmaCup22Dataset(data_dir=str(dirs['raw']))
    train_meta, test_meta = dataset_handler.load_data()
    train_meta['original_index'] = train_meta.index

    if DEBUG:
        print("!!! DEBUG MODE: Using small subset !!!")
        uq = train_meta['quarter'].unique()
        if len(uq) >= 2:
            train_meta = pd.concat([train_meta[train_meta['quarter'] == uq[0]].head(100), train_meta[train_meta['quarter'] == uq[1]].head(100)])
        else: train_meta = train_meta.head(200)
        test_meta = test_meta.head(50)
        
    crops_dir = dirs['processed'] / 'crops_train'
    if not list(crops_dir.glob("*.jpg")): crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')
    
    bg_crops_dir = dirs['processed'] / 'crops_bg'
    bg_csv_path = dirs['processed'] / 'train_meta_background.csv'
    if not bg_csv_path.exists(): bg_df = generate_background_samples(train_meta, dirs['raw'], bg_crops_dir, 1, -1)
    else: bg_df = pd.read_csv(bg_csv_path)
    bg_df['original_index'] = -1
    full_train_df = pd.concat([train_meta, bg_df], axis=0, ignore_index=True)

    # Transforms (Same)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomRotation(15), 
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    ti, vi = next(sgkf.split(full_train_df.index, full_train_df['label_id'].astype(str), groups=full_train_df['quarter']))
    train_df, val_df = full_train_df.iloc[ti], full_train_df.iloc[vi]
    
    crop_dirs = {'train': crops_dir, 'bg': bg_crops_dir}
    train_ds = MixedImageDataset(train_df, crop_dirs, train_transform)
    val_ds = MixedImageDataset(val_df, crop_dirs, val_transform, mode='validation')
    batch_size = 32 if DEBUG else 256
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    model = AtmaCupModel(12, pretrained=True, freeze_backbone=True, use_arcface=False, use_embedding_head=False)
    model.to(device)
    
    # [MODIFIED] Initialize EMA
    ema = ModelEMA(model, decay=0.999, device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler(enabled=use_amp)
    
    best_score_ema = 0.0
    exp_output_dir = dirs['output'] / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = dirs['output'] / exp_name / 'models' / f"{exp_name}_best.pth"
    best_model_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # [MODIFIED] Update EMA
            ema.update(model)
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader.dataset)
        
        # Validate Normal
        val_loss, metrics = validate(model, val_loader, device, use_amp, criterion)
        # Validate EMA
        val_loss_ema, metrics_ema = validate(ema.module, val_loader, device, use_amp, criterion, desc="Val(EMA)")
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  [Normal] Loss: {val_loss:.4f} | F1: {metrics['macro_f1_all']:.4f} | Player F1: {metrics['macro_f1_player']:.4f}")
        print(f"  [EMA   ] Loss: {val_loss_ema:.4f} | F1: {metrics_ema['macro_f1_all']:.4f} | Player F1: {metrics_ema['macro_f1_player']:.4f}")
        
        # Use EMA score for checkpoint
        if metrics_ema["macro_f1_all"] > best_score_ema:
            best_score_ema = metrics_ema["macro_f1_all"]
            torch.save(ema.module.state_dict(), best_model_path)
            
        scheduler.step(metrics_ema["macro_f1_all"])

    print(f"Best Val F1 (EMA): {best_score_ema:.4f}")
    save_results({'val_score_ema': best_score_ema}, str(exp_output_dir), exp_name)

if __name__ == "__main__":
    main()
