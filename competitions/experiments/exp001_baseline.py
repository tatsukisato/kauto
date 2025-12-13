"""
Experiment 001: Baseline Model
Simple LightGBM model using only bbox position features
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import AtmaCup22Dataset
from validation import create_time_split, evaluate_predictions
from model import LGBMClassifier
from utils import print_experiment_info, save_results, create_submission
from experiment_logger import log_experiment_to_markdown, log_experiment_to_json


def main():
    # Experiment configuration
    EXP_NAME = "exp001_baseline"
    DESCRIPTION = "Baseline LightGBM model with bbox position features only"
    
    print_experiment_info(EXP_NAME, DESCRIPTION)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "output" / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    dataset = AtmaCup22Dataset(data_dir=str(data_dir))
    train_meta, test_meta = dataset.load_data()
    
    # 2. Create train/validation split
    print("\n[2/6] Creating train/validation split...")
    # Note: Train data only contains Q1 and Q2, not Q4
    # So we use Q2's later sessions for validation
    train_df = train_meta[train_meta['quarter'].str.startswith('Q1')].copy()
    val_df = train_meta[train_meta['quarter'].str.startswith('Q2')].copy()
    
    print(f"Train split (Q1): {len(train_df)} samples")
    print(f"Validation split (Q2): {len(val_df)} samples")
    
    # 3. Create features
    print("\n[3/6] Creating features...")
    X_train = dataset.create_features(train_df)
    y_train = train_df['label_id']
    
    X_val = dataset.create_features(val_df)
    y_val = val_df['label_id']
    
    X_test = dataset.get_test_data()
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature names: {list(X_train.columns)}")
    
    # 4. Train model
    print("\n[4/6] Training model...")
    model = LGBMClassifier(
        params={
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        },
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    model.train(
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values
    )
    
    # Save model
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path))
    
    # 5. Evaluate on validation set
    print("\n[5/6] Evaluating on validation set...")
    val_preds = model.predict(X_val.values)
    
    results = evaluate_predictions(
        y_val.values,
        val_preds,
        label_names={i: f"Player_{i}" for i in range(11)}
    )
    
    # Save results
    save_results(results, str(output_dir), EXP_NAME)
    
    # 6. Create submission
    print("\n[6/6] Creating submission...")
    test_preds = model.predict(X_test.values)
    
    submission_path = base_dir / "submissions" / f"{EXP_NAME}.csv"
    create_submission(
        test_preds.tolist(),
        str(submission_path),
        test_meta
    )
    
    print("\n" + "=" * 80)
    print(f"Experiment {EXP_NAME} completed!")
    print(f"Validation Macro F1: {results['macro_f1']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Submission saved to: {submission_path}")
    print("=" * 80)
    
    # Log experiment automatically
    print("\n[Logging] Recording experiment...")
    
    # Prepare config for logging
    config = {
        'model': 'LightGBM (多クラス分類)',
        'features': list(X_train.columns),
        'validation': 'Q1を学習、Q2をバリデーション',
        'params': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'num_boost_round': 1000,
            'early_stopping_rounds': 50,
        }
    }
    
    # Add additional info to results
    results['best_iteration'] = model.model.best_iteration
    results['train_samples'] = len(X_train)
    results['val_samples'] = len(X_val)
    
    # Observations
    observations = [
        "位置情報のみでは限界: Macro F1が0.17と低い",
        f"Player 4が識別しやすい: F1={results['per_class_f1'].get(4, 0):.2f}で最高スコア",
        "クラス不均衡の影響: Q1/Q2で登場しない選手のF1が0.0",
        f"早期停止: {model.model.best_iteration}イテレーションで停止",
    ]
    
    # Next steps
    next_steps = [
        "画像特徴の追加",
        "topカメラデータの活用",
        "時系列特徴（前後フレームとの差分）",
        "ディープラーニングモデルの検討",
    ]
    
    # Log to markdown
    log_experiment_to_markdown(
        exp_name=EXP_NAME,
        description=DESCRIPTION,
        config=config,
        results=results,
        observations=observations,
        next_steps=next_steps,
        log_file=str(base_dir / "docs" / "EXPERIMENTS.md")
    )
    
    # Log to JSON
    log_experiment_to_json(
        exp_name=EXP_NAME,
        config=config,
        results=results,
        log_file=str(base_dir / "output" / "experiment_history.json")
    )


if __name__ == "__main__":
    main()
