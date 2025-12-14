
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
import numpy as np

# Add project root to path to import src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import setup_directories, save_results, create_submission, print_experiment_info, crop_and_save_images
from src.image_dataset import ImageDataset
from src.cnn_model import SimpleCNN
from src.dataset import AtmaCup22Dataset

def main():
    exp_name = "exp002_image_baseline"
    description = "Image-based baseline with ResNet18 (Frozen backbone). Train on Q1, Val on Q2."
    print_experiment_info(exp_name, description)
    
    dirs = setup_directories()
    
    # 1. Load Data
    dataset_handler = AtmaCup22Dataset(data_dir=str(dirs['raw']))
    train_meta, test_meta = dataset_handler.load_data()
    
    # 2. Prepare Data Splitting
    # Train: Q1, Val: Q2
    # Filter by quarter string
    train_df = train_meta[train_meta['quarter'].str.contains('Q1')].copy()
    val_df = train_meta[train_meta['quarter'].str.contains('Q2')].copy()
    
    print(f"Train set (Q1): {len(train_df)}")
    print(f"Val set (Q2): {len(val_df)}")
    
    # 3. Check/Generate Cropped Images
    # We store generated crops in data/processed/crops_train
    crops_dir = dirs['processed'] / 'crops_train'
    
    # Check if crops exist for all training data (Q1+Q2)
    # We use the length of original train_meta because indices are based on it
    # and we want to ensure all potential images are processed if we change split later
    if not crops_dir.exists() or len(list(crops_dir.glob("*.jpg"))) < len(train_meta) * 0.9: 
        # *0.9 allows for some missing/failed crops, but ideally should be full.
        # Let's just generate if dir doesn't exist or seems empty
        print(f"Generating cropped images to {crops_dir}...")
        crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')
    else:
        print(f"Using existing cropped images in {crops_dir}")
    
    # 4. Transforms
    # cv2 reads as numpy array (H, W, C). ToPILImage converts to PIL.
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
    
    # 5. Datasets & Loaders
    train_dataset = ImageDataset(train_df, str(crops_dir), transform=train_transform, mode='train')
    val_dataset = ImageDataset(val_df, str(crops_dir), transform=val_transform, mode='validation')
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 6. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleCNN(num_classes=11, pretrained=True, freeze_backbone=True)
    model.to(device)
    
    # 7. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 10
    best_f1 = 0.0
    # Create experiment specific output directory
    exp_output_dir = dirs['output'] / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update model path to be inside experiment directory
    model_dir = exp_output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    best_model_path = model_dir / f"{exp_name}_best.pth"
    
    # 8. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Macro F1: {macro_f1:.4f}")
        
        scheduler.step(macro_f1)
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
            
    # 9. Test Prediction
    print("Starting prediction on Test Data (Q4)...")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found, using last epoch model")
        
    model.eval()
    
    # Test dataset
    # For test, image_dir should be 'data/raw' as rel_paths are relative to it
    test_dataset = ImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    test_preds = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            
    # 10. Create Submission
    sub_path = dirs['submissions'] / f"submission_{exp_name}.csv"
    create_submission(test_preds, str(sub_path), test_meta)
    
    # Save results
    save_results({
        'best_val_f1': best_f1,
        'epochs': epochs,
        'config': {
            'backbone': 'resnet18',
            'img_size': 224,
            'batch_size': batch_size,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset) 
        }
    }, str(exp_output_dir), exp_name)

if __name__ == "__main__":
    main()
