{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ce8bcf1f",
   "metadata": {
    "_cell_guid": "85f5629e-421c-4f74-ae17-cbfa9cfddff0",
    "_uuid": "b2c1ba57-5dae-4299-96b5-ee4c9912ba12",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.002241,
     "end_time": "2025-12-15T05:22:16.759341",
     "exception": false,
     "start_time": "2025-12-15T05:22:16.757100",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exp004: Stratified Group K-Fold Validation Strategy\n",
    "\n",
    "## Objective\n",
    "Refine the validation strategy to resolve BOTH \"Class Distribution Mismatch\" and \"Temporal Leakage\".\n",
    "\n",
    "## Strategy: Stratified Group K-Fold\n",
    "- **Problem 1 (Class Balance)**: Random split keeps classes balanced but mixes adjacent frames (Leakage).\n",
    "- **Problem 2 (Leakage)**: Simple GroupKFold by 'quarter' prevents leakage but might lead to missing classes if a player only appears in specific quarters.\n",
    "- **Solution**: **StratifiedGroupKFold** with `groups=quarter`.\n",
    "  - **Grouping**: Ensures all frames from the same quarter (e.g., \"Q1-001\") are in the same fold. This prevents the model from memorizing the specific sequence/background of a chunk.\n",
    "  - **Stratification**: Attempts to balance class ratios across folds by selecting combination of groups.\n",
    "\n",
    "## Groups\n",
    "- We use the `quarter` column (e.g., \"Q1-001\", \"Q2-005\") as the group identifier."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "55bbc04e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-12-15T05:22:16.764329Z",
     "iopub.status.busy": "2025-12-15T05:22:16.763639Z",
     "iopub.status.idle": "2025-12-15T05:22:17.483368Z",
     "shell.execute_reply": "2025-12-15T05:22:17.482289Z"
    },
    "papermill": {
     "duration": 0.723872,
     "end_time": "2025-12-15T05:22:17.484977",
     "exception": false,
     "start_time": "2025-12-15T05:22:16.761105",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cloning into 'kauto'...\r\n",
      "remote: Enumerating objects: 142, done.\u001b[K\r\n",
      "remote: Counting objects: 100% (142/142), done.\u001b[K\r\n",
      "remote: Compressing objects: 100% (112/112), done.\u001b[K\r\n",
      "remote: Total 142 (delta 33), reused 125 (delta 19), pack-reused 0 (from 0)\u001b[K\r\n",
      "Receiving objects: 100% (142/142), 210.68 KiB | 8.10 MiB/s, done.\r\n",
      "Resolving deltas: 100% (33/33), done.\r\n"
     ]
    }
   ],
   "source": [
    "!git clone https://github.com/tatsukisato/kauto.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "21741f5c",
   "metadata": {
    "_cell_guid": "3d72eed4-6c26-48dc-aef0-433faf8777ec",
    "_uuid": "489605b4-9fa6-4ad9-a17e-589b768103a1",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-15T05:22:17.491227Z",
     "iopub.status.busy": "2025-12-15T05:22:17.490633Z",
     "iopub.status.idle": "2025-12-15T05:22:28.161841Z",
     "shell.execute_reply": "2025-12-15T05:22:28.161200Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 10.675697,
     "end_time": "2025-12-15T05:22:28.163131",
     "exception": false,
     "start_time": "2025-12-15T05:22:17.487434",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.amp import autocast, GradScaler\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import transforms\n",
    "from tqdm import tqdm\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.model_selection import StratifiedGroupKFold\n",
    "import numpy as np\n",
    "\n",
    "# Detect environment\n",
    "IS_KAGGLE = os.path.exists(\"/kaggle/input\")\n",
    "ROOT_DIR = Path(\"/kaggle/working/kauto/competitions\") if IS_KAGGLE else Path(__file__).resolve().parents[1]\n",
    "DATA_DIR = Path(\"/kaggle/input/atmacup22\") if IS_KAGGLE else Path(__file__).resolve().parents[1]\n",
    "\n",
    "sys.path.append(str(ROOT_DIR))\n",
    "\n",
    "try:\n",
    "    from src.utils import setup_directories, save_results, create_submission, print_experiment_info, crop_and_save_images\n",
    "    from src.image_dataset import ImageDataset\n",
    "    from src.cnn_model import SimpleCNN\n",
    "    from src.dataset import AtmaCup22Dataset\n",
    "except ImportError:\n",
    "    print(\"Warning: Custom modules not found.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "51aee302",
   "metadata": {
    "_cell_guid": "c5b77c0c-bab0-498a-b884-a70d3e62b608",
    "_uuid": "568fb8fa-6d25-40b5-af39-728880397308",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-12-15T05:22:28.168385Z",
     "iopub.status.busy": "2025-12-15T05:22:28.168005Z",
     "iopub.status.idle": "2025-12-15T05:29:41.607579Z",
     "shell.execute_reply": "2025-12-15T05:29:41.606596Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 433.491522,
     "end_time": "2025-12-15T05:29:41.656737",
     "exception": false,
     "start_time": "2025-12-15T05:22:28.165215",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "================================================================================\n",
      "Train data shape: (24920, 9)\n",
      "Test data shape: (9223, 9)\n",
      "Using device: cuda\n",
      "\n",
      "==================== Hold-out Validation (20%) ====================\n",
      "Train size: 19880, Val size: 5040\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Downloading: \"https://download.pytorch.org/models/resnet18-f37072fd.pth\" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth\n",
      "100%|██████████| 44.7M/44.7M [00:00<00:00, 195MB/s]\n",
      "Ep 1/8: 100%|██████████| 78/78 [00:51<00:00,  1.51it/s, loss=1.23]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.3797, Val F1: 0.5401\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 2/8: 100%|██████████| 78/78 [00:39<00:00,  1.97it/s, loss=0.969]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.2206, Val F1: 0.6016\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 3/8: 100%|██████████| 78/78 [00:40<00:00,  1.93it/s, loss=1.17]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.1963, Val F1: 0.6031\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 4/8: 100%|██████████| 78/78 [00:39<00:00,  1.98it/s, loss=1.01]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.1480, Val F1: 0.6117\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 5/8: 100%|██████████| 78/78 [00:39<00:00,  1.96it/s, loss=0.942]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.1181, Val F1: 0.6222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 6/8: 100%|██████████| 78/78 [00:39<00:00,  1.97it/s, loss=0.861]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.1120, Val F1: 0.6338\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 7/8: 100%|██████████| 78/78 [00:39<00:00,  1.96it/s, loss=0.792]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.0747, Val F1: 0.6489\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Ep 8/8: 100%|██████████| 78/78 [00:39<00:00,  2.00it/s, loss=0.82]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Val Loss: 1.0564, Val F1: 0.6604\n",
      "Best Val F1: 0.6604\n",
      "Submission saved to /kaggle/working/submissions/submission_exp004_stratified_group_validation.csv\n",
      "Submission shape: (9223, 1)\n",
      "Label distribution:\n",
      "label_id\n",
      "0       44\n",
      "1     1127\n",
      "2     1218\n",
      "3      653\n",
      "4     1162\n",
      "5      699\n",
      "6      789\n",
      "7      919\n",
      "8      965\n",
      "9      809\n",
      "10     838\n",
      "Name: count, dtype: int64\n",
      "Results saved to /kaggle/working/output/exp004_stratified_group_validation/exp004_stratified_group_validation_results.json\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    exp_name = \"exp004_stratified_group_validation\"\n",
    "    description = \"Stratified Group K-Fold (k=5, groups=quarter) to resolve class mismatch AND temporal leakage.\"\n",
    "    \n",
    "    DEBUG = not IS_KAGGLE\n",
    "    N_FOLDS = 5\n",
    "    EPOCHS = 2 if DEBUG else 8\n",
    "    \n",
    "    # Setup Directories\n",
    "    current_dir = Path.cwd()\n",
    "    if IS_KAGGLE:\n",
    "        data_input_dir = Path(\"/kaggle/input/atmacup22\") \n",
    "        base_output_dir = Path(\"/kaggle/working\")\n",
    "    else:\n",
    "        data_input_dir = ROOT_DIR\n",
    "        base_output_dir = ROOT_DIR\n",
    "\n",
    "    dirs = setup_directories(base_dir=str(base_output_dir), data_dir=str(data_input_dir))\n",
    "    print_experiment_info(exp_name, description)\n",
    "    \n",
    "    # 1. Load Data\n",
    "    raw_dir = dirs['raw']\n",
    "    dataset_handler = AtmaCup22Dataset(data_dir=str(raw_dir))\n",
    "    train_meta, test_meta = dataset_handler.load_data()\n",
    "    \n",
    "    # Define Groups\n",
    "    groups = train_meta['quarter']\n",
    "    \n",
    "    if DEBUG:\n",
    "        print(\"!!! DEBUG MODE: Using small subset !!!\")\n",
    "        # Ensure we have multiple groups (quarters) for StratifiedGroupKFold\n",
    "        # Simply taking head(200) might result in only 1 group (e.g. Q1-000)\n",
    "        unique_quarters = train_meta['quarter'].unique()\n",
    "        if len(unique_quarters) >= 2:\n",
    "            # Take first 100 rows from first 2 quarters\n",
    "            q1_df = train_meta[train_meta['quarter'] == unique_quarters[0]].head(100)\n",
    "            q2_df = train_meta[train_meta['quarter'] == unique_quarters[1]].head(100)\n",
    "            train_meta = pd.concat([q1_df, q2_df])\n",
    "        else:\n",
    "            # Fallback if only 1 quarter exists (unlikely for full data)\n",
    "            train_meta = train_meta.iloc[:200]\n",
    "            \n",
    "        test_meta = test_meta.iloc[:50]\n",
    "        # Update groups based on new train_meta\n",
    "        groups = train_meta['quarter']\n",
    "        N_FOLDS = 2\n",
    "    \n",
    "    # 2. Check Crops\n",
    "    crops_dir = dirs['processed'] / 'crops_train'\n",
    "    current_crops = list(crops_dir.glob(\"*.jpg\")) if crops_dir.exists() else []\n",
    "    if len(current_crops) < len(train_meta) * 0.9: \n",
    "        print(f\"Generating crops to {crops_dir}...\")\n",
    "        crop_and_save_images(train_meta, dirs['raw'], crops_dir, mode='train')\n",
    "    \n",
    "    # 3. Transforms\n",
    "    train_transform = transforms.Compose([\n",
    "        transforms.ToPILImage(), \n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.RandomHorizontalFlip(p=0.5),\n",
    "        transforms.ColorJitter(brightness=0.2, contrast=0.2),\n",
    "        transforms.RandomRotation(10),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "    ])\n",
    "    \n",
    "    val_transform = transforms.Compose([\n",
    "        transforms.ToPILImage(),\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "    ])\n",
    "    \n",
    "    # 4. Stratified Group Split (Hold-out)\n",
    "    # Use StratifiedGroupKFold with n_splits=5 to get a 80:20 split, \n",
    "    # but strictly run only the FIRST fold to save time.\n",
    "    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    \n",
    "    # Prepare arrays for OOF (Partial) and Test Predictions\n",
    "    # Note: OOF will only be filled for the validation set of the 1st fold\n",
    "    oof_preds = np.zeros(len(train_meta)) - 1\n",
    "    \n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\")\n",
    "    use_amp = device.type == \"cuda\"\n",
    "    print(f\"Using device: {device}\")\n",
    "    \n",
    "    # Create experiment output dir\n",
    "    exp_output_dir = dirs['output'] / exp_name\n",
    "    exp_output_dir.mkdir(parents=True, exist_ok=True)\n",
    "    model_dir = exp_output_dir / 'models'\n",
    "    model_dir.mkdir(exist_ok=True)\n",
    "\n",
    "    # 5. Training Loop (Single Fold)\n",
    "    X = train_meta.index.values \n",
    "    y = train_meta['label_id'].values\n",
    "    \n",
    "    # Get just the first split\n",
    "    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))\n",
    "    \n",
    "    print(f\"\\n{'='*20} Hold-out Validation (20%) {'='*20}\")\n",
    "    \n",
    "    train_df_fold = train_meta.iloc[train_idx]\n",
    "    val_df_fold = train_meta.iloc[val_idx]\n",
    "    \n",
    "    print(f\"Train size: {len(train_df_fold)}, Val size: {len(val_df_fold)}\")\n",
    "    \n",
    "    # Datasets\n",
    "    train_dataset = ImageDataset(train_df_fold, str(crops_dir), transform=train_transform, mode='train')\n",
    "    val_dataset = ImageDataset(val_df_fold, str(crops_dir), transform=val_transform, mode='validation')\n",
    "\n",
    "    batch_size = 256\n",
    "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)\n",
    "    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)\n",
    "    \n",
    "    # Model\n",
    "    model = SimpleCNN(num_classes=11, pretrained=True, freeze_backbone=True)\n",
    "    model.to(device)\n",
    "    \n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.Adam(model.parameters(), lr=4e-3)\n",
    "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)\n",
    "    scaler = GradScaler(enabled=use_amp)\n",
    "    \n",
    "    best_score = 0.0\n",
    "    best_model_path = model_dir / f\"{exp_name}_best.pth\"\n",
    "    \n",
    "    for epoch in range(EPOCHS):\n",
    "        model.train()\n",
    "        train_loss = 0.0\n",
    "        \n",
    "        pbar = tqdm(train_loader, desc=f\"Ep {epoch+1}/{EPOCHS}\")\n",
    "        for images, labels in pbar:\n",
    "            images, labels = images.to(device), labels.to(device)\n",
    "            \n",
    "            optimizer.zero_grad(set_to_none=True)\n",
    "            with autocast(device_type=\"cuda\", enabled=use_amp):\n",
    "                outputs = model(images)\n",
    "                loss = criterion(outputs, labels)\n",
    "            scaler.scale(loss).backward()\n",
    "            scaler.step(optimizer)\n",
    "            scaler.update()\n",
    "            \n",
    "            train_loss += loss.item() * images.size(0)\n",
    "            pbar.set_postfix({'loss': loss.item()})\n",
    "        \n",
    "        train_loss /= len(train_dataset)\n",
    "        \n",
    "        # Validation\n",
    "        model.eval()\n",
    "        val_loss = 0.0\n",
    "        fold_preds = []\n",
    "        fold_labels = []\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            for images, labels in val_loader:\n",
    "                images, labels = images.to(device), labels.to(device)\n",
    "                with autocast(device_type=\"cuda\", enabled=use_amp):\n",
    "                    outputs = model(images)\n",
    "                    loss = criterion(outputs, labels)\n",
    "                val_loss += loss.item() * images.size(0)\n",
    "                \n",
    "                preds = torch.argmax(outputs, dim=1).cpu().numpy()\n",
    "                fold_preds.extend(preds)\n",
    "                fold_labels.extend(labels.cpu().numpy())\n",
    "        \n",
    "        val_loss /= len(val_dataset)\n",
    "        macro_f1 = f1_score(fold_labels, fold_preds, average='macro')\n",
    "        \n",
    "        print(f\"  Val Loss: {val_loss:.4f}, Val F1: {macro_f1:.4f}\")\n",
    "        scheduler.step(macro_f1)\n",
    "        \n",
    "        if macro_f1 > best_score:\n",
    "            best_score = macro_f1\n",
    "            torch.save(model.state_dict(), best_model_path)\n",
    "    \n",
    "    print(f\"Best Val F1: {best_score:.4f}\")\n",
    "    \n",
    "    # Load best model for inference\n",
    "    model.load_state_dict(torch.load(best_model_path, map_location=device))\n",
    "    model.eval()\n",
    "    \n",
    "    # 1. Fill OOF (Partial)\n",
    "    val_preds = []\n",
    "    with torch.no_grad():\n",
    "        for images, labels in val_loader:\n",
    "            with autocast(device_type=\"cuda\", enabled=use_amp):\n",
    "                images = images.to(device)\n",
    "                outputs = model(images)\n",
    "            preds = torch.argmax(outputs, dim=1).cpu().numpy()\n",
    "            val_preds.extend(preds)\n",
    "    \n",
    "    # Since we are not looping, just assign directly\n",
    "    oof_preds[val_idx] = val_preds\n",
    "\n",
    "    # 2. Predict on Test\n",
    "    test_dataset = ImageDataset(test_meta, str(dirs['raw']), transform=val_transform, mode='test')\n",
    "    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)\n",
    "    \n",
    "    final_test_preds = []\n",
    "    with torch.no_grad():\n",
    "        for images in test_loader:\n",
    "            images = images.to(device)\n",
    "            with autocast(device_type=\"cuda\", enabled=use_amp):\n",
    "                outputs = model(images)\n",
    "            preds = torch.argmax(outputs, dim=1).cpu().numpy()\n",
    "            final_test_preds.extend(preds)\n",
    "\n",
    "    # Save OOF\n",
    "    oof_df = train_meta.copy()\n",
    "    oof_df['pred_label_id'] = oof_preds\n",
    "    # Only save rows that were in validation\n",
    "    oof_df_val = oof_df.iloc[val_idx]\n",
    "    oof_df_val.to_csv(exp_output_dir / 'oof_predictions_val_only.csv', index=False)\n",
    "    \n",
    "    # Create Submission\n",
    "    sub_path = dirs['submissions'] / f\"submission_{exp_name}.csv\"\n",
    "    create_submission(final_test_preds, str(sub_path), test_meta)\n",
    "    \n",
    "    # Save Experiment Info\n",
    "    save_results({\n",
    "        'val_score': best_score,\n",
    "        'config': {\n",
    "            'validation': 'stratified_group_holdout_20pct',\n",
    "            'epochs': EPOCHS,\n",
    "            'backbone': 'resnet18'\n",
    "        }\n",
    "    }, str(exp_output_dir), exp_name)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62bf03bb",
   "metadata": {
    "papermill": {
     "duration": 0.049648,
     "end_time": "2025-12-15T05:29:41.754772",
     "exception": false,
     "start_time": "2025-12-15T05:29:41.705124",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "gpu",
   "dataSources": [
    {
     "datasetId": 9021091,
     "sourceId": 14153731,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31193,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 451.23476,
   "end_time": "2025-12-15T05:29:44.559050",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-12-15T05:22:13.324290",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
