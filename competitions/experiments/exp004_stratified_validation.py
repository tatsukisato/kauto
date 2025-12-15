# %% [markdown]
# # Exp004: Stratified Group K-Fold Validation Strategy
# 
# ## Objective
# Refine the validation strategy to resolve BOTH "Class Distribution Mismatch" and "Temporal Leakage".
# 
# ## Strategy: Stratified Group K-Fold
# - **Problem 1 (Class Balance)**: Random split keeps classes balanced but mixes adjacent frames (Leakage).
# - **Problem 2 (Leakage)**: Simple GroupKFold by 'quarter' prevents leakage but might lead to missing classes if a player only appears in specific quarters.
# - **Solution**: **StratifiedGroupKFold** with `groups=quarter`.
#   - **Grouping**: Ensures all frames from the same quarter (e.g., "Q1-001") are in the same fold. This prevents the model from memorizing the specific sequence/background of a chunk.
#   - **Stratification**: Attempts to balance class ratios across folds by selecting combination of groups.
# 
# ## Groups
# - We use the `quarter` column (e.g., "Q1-001", "Q2-005") as the group identifier.

# %% [code]
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/input")
ROOT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path(__file__).resolve().parents[1]
DATA_DIR = Path("/kaggle/input") if IS_KAGGLE else Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

try:
    from src.utils import setup_directories, save_results, create_submission, print_experiment_info, crop_and_save_images
    from src.image_dataset import ImageDataset
    from src.cnn_model import SimpleCNN
    from src.dataset import AtmaCup22Dataset
except ImportError:
    print("Warning: Custom modules not found.")

# %% [code]
def main():
    exp_name = "exp004_stratified_group_validation"
    description = "Stratified Group K-Fold (k=5, groups=quarter) to resolve class mismatch AND temporal leakage."
    
    DEBUG = not IS_KAGGLE
    N_FOLDS = 5
    EPOCHS = 2 if DEBUG else 8
    
    # Setup Directories
    current_dir = Path.cwd()
    if IS_KAGGLE:
        data_input_dir = Path("/kaggle/input/atmacup22") 
        base_output_dir = Path("/kaggle/working")
    else:
        data_input_dir = ROOT_DIR
        base_output_dir = ROOT_DIR

    dirs = setup_directories(base_dir=str(base_output_dir), data_dir=str(data_input_dir))
    print_experiment_info(exp_name, description)
    
    # 1. Load Data
    raw_dir = dirs['raw']
    dataset_handler = AtmaCup22Dataset(data_dir=str(raw_dir))
    train_meta, test_meta = dataset_handler.load_data()
    
    # Define Groups
    groups = train_meta['quarter']
    
    if DEBUG:
        print("!!! DEBUG MODE: Using small subset !!!")
        # Ensure we have multiple groups (quarters) for StratifiedGroupKFold
        # Simply taking head(200) might result in only 1 group (e.g. Q1-000)
        unique_quarters = train_meta['quarter'].unique()
        if len(unique_quarters) >= 2:
            # Take first 100 rows from first 2 quarters
            q1_df = train_meta[train_meta['quarter'] == unique_quarters[0]].head(100)
            q2_df = train_meta[train_meta['quarter'] == unique_quarters[1]].head(100)
            train_meta = pd.concat([q1_df, q2_df])
        else:
            # Fallback if only 1 quarter exists (unlikely for full data)
            train_meta = train_meta.iloc[:200]
            
        test_meta = test_meta.iloc[:50]
        # Update groups based on new train_meta
        groups = train_meta['quarter']
        N_FOLDS = 2
    
    # 2. Check Crops
    crops_dir = dirs['processed'] / 'crops_train'
    current_crops = list(crops_dir.glob("*.jpg")) if crops_dir.exists() else []
    if len(current_crops) < len(train_meta) * 0.9: 
        print(f"Generating crops to {crops_dir}...")
        crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')
    
    # 3. Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Stratified Group Split (Hold-out)
    # Use StratifiedGroupKFold with n_splits=5 to get a 80:20 split, 
    # but strictly run only the FIRST fold to save time.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prepare arrays for OOF (Partial) and Test Predictions
    # Note: OOF will only be filled for the validation set of the 1st fold
    oof_preds = np.zeros(len(train_meta)) - 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment output dir
    exp_output_dir = dirs['output'] / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_output_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    # 5. Training Loop (Single Fold)
    X = train_meta.index.values 
    y = train_meta['label_id'].values
    
    # Get just the first split
    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))
    
    print(f"\n{'='*20} Hold-out Validation (20%) {'='*20}")
    
    train_df_fold = train_meta.iloc[train_idx]
    val_df_fold = train_meta.iloc[val_idx]
    
    print(f"Train size: {len(train_df_fold)}, Val size: {len(val_df_fold)}")
    
    # Datasets
    train_dataset = ImageDataset(train_df_fold, str(crops_dir), transform=train_transform, mode='train')
    val_dataset = ImageDataset(val_df_fold, str(crops_dir), transform=val_transform, mode='validation')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Model
    model = SimpleCNN(num_classes=11, pretrained=True, freeze_backbone=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_score = 0.0
    best_model_path = model_dir / f"{exp_name}_best.pth"
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        macro_f1 = f1_score(fold_labels, fold_preds, average='macro')
        
        print(f"  Val Loss: {val_loss:.4f}, Val F1: {macro_f1:.4f}")
        scheduler.step(macro_f1)
        
        if macro_f1 > best_score:
            best_score = macro_f1
            torch.save(model.state_dict(), best_model_path)
    
    print(f"Best Val F1: {best_score:.4f}")
    
    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # 1. Fill OOF (Partial)
    val_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
    
    # Since we are not looping, just assign directly
    oof_preds[val_idx] = val_preds

    # 2. Predict on Test
    test_dataset = ImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    final_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            final_test_preds.extend(preds)

    # Save OOF
    oof_df = train_meta.copy()
    oof_df['pred_label_id'] = oof_preds
    # Only save rows that were in validation
    oof_df_val = oof_df.iloc[val_idx]
    oof_df_val.to_csv(exp_output_dir / 'oof_predictions_val_only.csv', index=False)
    
    # Create Submission
    sub_path = dirs['submissions'] / f"submission_{exp_name}.csv"
    create_submission(final_test_preds, str(sub_path), test_meta)
    
    # Save Experiment Info
    save_results({
        'val_score': best_score,
        'config': {
            'validation': 'stratified_group_holdout_20pct',
            'epochs': EPOCHS,
            'backbone': 'resnet18'
        }
    }, str(exp_output_dir), exp_name)

if __name__ == "__main__":
    main()
