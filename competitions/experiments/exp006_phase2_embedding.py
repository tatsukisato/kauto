# === Added for exp006 improvement ===
# Exp006: Improvements Phase 2 Step 3 (Embedding Head Verification)
# Base: Exp006 Phase 2 Step 1
# Improvements:
# 1. Enable use_embedding_head=True in AtmaCupModel
# 2. Verify training with BN->FC->BN structure

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
        create_submission,
        print_experiment_info,
        crop_and_save_images,
    )
    from src.dataset import AtmaCup22Dataset, MixedImageDataset
    from src.models import AtmaCupModel
    from src.generate_background import generate_background_samples
    from src.image_dataset import ImageDataset as StandardImageDataset  # 追加: test 用 Dataset
    from src.metrics import compute_evaluation_metrics
except ImportError:
    print("Warning: Custom modules not found.")

def main():
    exp_name = "exp006_phase2_embedding"
    description = "Phase 2 Step 3: Embedding Head Verification. use_embedding_head=True."
    
    DEBUG = not IS_KAGGLE
    EPOCHS = 2 if DEBUG else 12 
    # ArcFace toggle: set True to enable ArcFace head/softmax-margin behavior.
    # 注意: ArcFace 層は内部で F.normalize を行い、s=30.0, m=0.5 を想定しています。
    # ArcFace に渡す入力は「L2 正規化 前 の embedding」を渡す（モデル内部で正規化される前提）。
    USE_ARCFACE = False
    
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
         print("Generating player crops...")
         crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')
    
    # 3. Check/Generate Background Crops
    bg_crops_dir = dirs['processed'] / 'crops_bg'
    bg_csv_path = dirs['processed'] / 'train_meta_background.csv'
    
    if not bg_csv_path.exists():
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
    
    groups = full_train_df['quarter']
    X = full_train_df.index
    y = full_train_df['label_id'].astype(str)
    
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
    
    # 6. Model (12 Classes) - EMBEDDING HEAD ENABLED
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}, AMP: {use_amp}")
    
    model = AtmaCupModel(
        num_classes=12,
        pretrained=True,
        freeze_backbone=True,
        use_arcface=USE_ARCFACE,    # フラグに従う
        use_embedding_head=True,    # BN->FC->BN 構造を確認
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # 学習率を大きすぎないように設定（1e-3 -> 3e-4 を推奨）
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
                # If ArcFace is enabled, pass targets during forward so ArcFace layer
                # can apply margin inside (model is expected to accept targets=labels).
                # Important: pass embedding BEFORE external normalization; model's ArcFace
                # layer will call F.normalize internally (s=30.0, m=0.5 assumed).
                if USE_ARCFACE:
                    outputs = model(images, targets=labels)
                else:
                    outputs = model(images)
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
                    # In eval, do not pass targets so model returns inference scores
                    if USE_ARCFACE:
                        outputs = model(images)  # returns scaled cosine (inference)
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        # Validation Metrics (delegated to src.metrics)
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        metrics = compute_evaluation_metrics(val_labels, val_preds, bg_label=11)
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  [Overall] F1: {metrics['macro_f1_all']:.4f}")
        print(f"  [Player ] F1: {metrics['macro_f1_player']:.4f}")
        print(f"  [BG Stats] Recall: {metrics['bg_recall']:.4f}, Precision: {metrics['bg_precision']:.4f} (FP={metrics['bg_fp']})")
        macro_f1_all = metrics["macro_f1_all"]
        macro_f1_player = metrics["macro_f1_player"]

        if macro_f1_all > best_score:
            best_score = macro_f1_all
            torch.save(model.state_dict(), best_model_path)
            
        scheduler.step(macro_f1_all)
        
    print(f"Best Val F1 (Overall): {best_score:.4f}")
    # --- Inference on Test and create submission ---
    try:
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            test_dataset = StandardImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            final_test_preds = []
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)
                    with autocast(device_type="cuda", enabled=use_amp):
                        outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds = np.where(preds == 11, -1, preds)  # map BG class back to -1
                    final_test_preds.extend(preds)

            sub_path = dirs['submissions'] / f"submission_{exp_name}.csv"
            create_submission(final_test_preds, str(sub_path), test_meta)
            print(f"Saved submission: {sub_path}")
        else:
            print("No best model found for inference; skipping test inference.")
    except Exception as e:
        print(f"Warning: Test inference failed: {e}")

    save_results({'val_score_overall': best_score, 'val_score_player': macro_f1_player}, str(exp_output_dir), exp_name)

if __name__ == "__main__":
    main()
