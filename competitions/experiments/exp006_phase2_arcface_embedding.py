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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from typing import Sized, cast

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
    from src.image_dataset import ImageDataset as StandardImageDataset
    from src.metrics import compute_evaluation_metrics
except ImportError:
    print("Warning: Custom modules not found.")

# (このファイルは単体実行用。既存 experiments のパターンに合わせています。)

def build_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def load_and_prepare_data(dirs, debug):
    ds = AtmaCup22Dataset(data_dir=str(dirs['raw']))
    train_meta, test_meta = ds.load_data()
    train_meta['original_index'] = train_meta.index

    if debug:
        uq = train_meta['quarter'].unique()
        if len(uq) >= 2:
            train_meta = pd.concat([
                train_meta[train_meta['quarter'] == uq[0]].head(100),
                train_meta[train_meta['quarter'] == uq[1]].head(100),
            ])
        else:
            train_meta = train_meta.head(200)
        test_meta = test_meta.head(50)

    crops_dir = dirs['processed'] / 'crops_train'
    if not list(crops_dir.glob("*.jpg")):
        print("Generating player crops...")
        crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')

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
    # 変更: train_meta を戻り値に追加
    return full_train_df, train_meta, test_meta, crops_dir, bg_crops_dir

def make_dataloaders(full_train_df, crops_dir, bg_crops_dir, train_transform, val_transform, debug, batch_size=256, num_workers=4):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = full_train_df.index.to_numpy()
    y = full_train_df['label_id'].astype(str).to_numpy()
    groups = full_train_df['quarter'].to_numpy()
    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))
    train_df_fold = full_train_df.iloc[train_idx]
    val_df_fold = full_train_df.iloc[val_idx]

    crop_dirs = {'train': crops_dir, 'bg': bg_crops_dir}
    train_dataset = MixedImageDataset(train_df_fold, crop_dirs, transform=train_transform, mode='train')
    val_dataset = MixedImageDataset(val_df_fold, crop_dirs, transform=val_transform, mode='validation')

    bsize = 32 if debug else batch_size
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset, val_dataset

def build_model(device, use_arcface=True, use_embedding_head=True):
    model = AtmaCupModel(
        num_classes=12,
        pretrained=True,
        freeze_backbone=True,
        use_arcface=use_arcface,
        use_embedding_head=use_embedding_head,
    )
    model.to(device)
    return model

def validate_and_metrics(model, val_loader, device, use_amp, criterion):
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

def train_loop(model, train_loader, val_loader, device, use_amp, criterion, optimizer, scheduler, scaler, exp_output_dir, best_model_path, epochs=12):
    best_score = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                # For ArcFace training, model(images, targets=labels) handled by model impl if needed
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader.dataset)

        val_loss, metrics = validate_and_metrics(model, val_loader, device, use_amp, criterion)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  [Val] Loss: {val_loss:.4f} | F1: {metrics['macro_f1_all']:.4f} | Player: {metrics['macro_f1_player']:.4f} | BG prec: {metrics['bg_precision']:.4f}")

        if metrics['macro_f1_all'] > best_score:
            best_score = metrics['macro_f1_all']
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(metrics['macro_f1_all'])
    return best_score

def infer_and_submit(model, best_model_path, test_meta, dirs, val_transform, batch_size, num_workers, device, use_amp, exp_name):
    try:
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            test_dataset = StandardImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            final_test_preds = []
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)
                    with autocast(device_type="cuda", enabled=use_amp):
                        outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds = np.where(preds == 11, -1, preds)
                    final_test_preds.extend(preds)
            sub_path = dirs['submissions'] / f"submission_{exp_name}.csv"
            create_submission(final_test_preds, str(sub_path), test_meta)
            print(f"Saved submission: {sub_path}")
        else:
            print("No best model found for inference; skipping test inference.")
    except Exception as e:
        print(f"Warning: Test inference failed: {e}")

def main():
    # 1) 実験名・説明・デバッグ設定の初期化
    exp_name = "exp006_phase2_arcface_embedding"
    description = "Phase2 D: ArcFace + Embedding Head (no EMA)"
    DEBUG = not IS_KAGGLE
    EPOCHS = 2 if DEBUG else 12

    # 2) パス/ディレクトリのセットアップ（Kaggle/ローカル共通）
    dirs = setup_directories(base_dir=str(Path("/kaggle/working") if IS_KAGGLE else ROOT_DIR), data_dir=str(DATA_DIR))
    print_experiment_info(exp_name, description)

    # 3) データ読み込みと（DEBUG時の）サブサンプリング
    #    - train_meta, test_meta を読み込み original_index を付与
    #    - DEBUG モードなら少量データに切り替え
    # 変更: train_meta を受け取るようにアンパック
    full_train_df, train_meta, test_meta, crops_dir, bg_crops_dir = load_and_prepare_data(dirs, DEBUG)

    # 4) プレイヤー crop と背景サンプルの確認・生成
    #    - crops_dir に画像がなければ crop を生成
    #    - 背景 CSV がなければ背景サンプルを生成
    crops_dir = dirs['processed'] / 'crops_train'
    if not list(crops_dir.glob("*.jpg")):
        print("Generating player crops...")
        crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')

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

    # 5) 画像変換（train/val）設定
    train_transform, val_transform = build_transforms()

    # 6) StratifiedGroupKFold によるホールドアウト分割（group=quarter）
    #    - train/val DataFrame を作成
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = full_train_df.index.to_numpy()
    y = full_train_df['label_id'].astype(str).to_numpy()
    groups = full_train_df['quarter'].to_numpy()
    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))
    train_df_fold = full_train_df.iloc[train_idx]
    val_df_fold = full_train_df.iloc[val_idx]

    # 7) Dataset / DataLoader の作成
    #    - MixedImageDataset を使って train/val loader を生成
    #    - batch_size, num_workers の設定
    crop_dirs = {'train': crops_dir, 'bg': bg_crops_dir}
    train_dataset = MixedImageDataset(train_df_fold, crop_dirs, transform=train_transform, mode='train')
    val_dataset = MixedImageDataset(val_df_fold, crop_dirs, transform=val_transform, mode='validation')

    bsize = 32 if DEBUG else 256
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False, num_workers=4)

    # 8) モデル / 損失 / 最適化器 / スケジューラ / AMP スケーラ の構築
    #    - AtmaCupModel を ArcFace + Embedding の組合せで構築
    #    - optimizer.lr は安定化のため小さめに設定
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = AtmaCupModel(
        num_classes=12,
        pretrained=True,
        freeze_backbone=True,
        use_arcface=True,
        use_embedding_head=True,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler(enabled=use_amp)

    # 9) 学習ループ（エポック毎）
    #    - train モードでバッチ学習（AMP あり）
    #    - バックプロパゲーションと optimizer.step(), scaler.update()
    #    - バリデーション実行 -> compute_evaluation_metrics を使用して指標算出
    #    - ベストモデル保存と scheduler.step() の更新
    best_score = 0.0
    exp_output_dir = dirs['output'] / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    best_model_path = model_dir / f"{exp_name}_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                # For ArcFace training, model(images, targets=labels) handled by model impl if needed
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(cast(Sized, train_loader.dataset))

        val_loss, metrics = validate_and_metrics(model, val_loader, device, use_amp, criterion)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  [Val] Loss: {val_loss:.4f} | F1: {metrics['macro_f1_all']:.4f} | Player: {metrics['macro_f1_player']:.4f} | BG prec: {metrics['bg_precision']:.4f}")

        if metrics['macro_f1_all'] > best_score:
            best_score = metrics['macro_f1_all']
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(metrics['macro_f1_all'])

    # 10) テスト推論と submission 作成
    #     - ベストモデルを読み込み、test 用 Dataset で推論
    #     - BG クラス(11) を -1 に戻して create_submission を呼ぶ
    try:
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            test_dataset = StandardImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=(32 if DEBUG else 256), shuffle=False, num_workers=4)
            final_test_preds = []
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)
                    with autocast(device_type="cuda", enabled=use_amp):
                        outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds = np.where(preds == 11, -1, preds)
                    final_test_preds.extend(preds)
            sub_path = dirs['submissions'] / f"submission_{exp_name}.csv"
            create_submission(final_test_preds, str(sub_path), test_meta)
            print(f"Saved submission: {sub_path}")
        else:
            print("No best model found for inference; skipping test inference.")
    except Exception as e:
        print(f"Warning: Test inference failed: {e}")

    # 11) 結果保存（save_results）
    save_results({'val_score_overall': best_score}, str(exp_output_dir), exp_name)


if __name__ == "__main__":
    main()