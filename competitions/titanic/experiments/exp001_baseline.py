import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from competitions.titanic.src.features.titanic_basic import TitanicBasicFeatures
from competitions.titanic.src.models.lgbm import LGBMModel

def run():
    # Setup Paths
    base_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = base_dir / "data" / "raw"
    # Experiment output goes to output/exp001_baseline
    exp_name = "exp001_baseline"
    output_dir = base_dir / "output" / exp_name
    processed_data_dir = output_dir / "data" # Save processed data specific to this exp
    
    print(f"Running Experiment: {exp_name}")
    print(f"Output Directory: {output_dir}")

    # 1. Feature Engineering
    print("--- Feature Engineering ---")
    fe = TitanicBasicFeatures(input_dir=raw_data_dir, output_dir=processed_data_dir)
    fe.run()
    
    # Load processed data
    X = pd.read_csv(processed_data_dir / "X_train.csv")
    y = pd.read_csv(processed_data_dir / "y_train.csv").values.ravel()
    X_test = pd.read_csv(processed_data_dir / "X_test.csv")

    # 2. Model Training
    print("--- Model Training ---")
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMModel(params=params, output_dir=output_dir / "models")
    oof_preds, scores = model.train(X, y)

    # 3. Prediction & Submission
    print("--- Prediction ---")
    test_preds = model.predict(X_test)
    
    # Create submission
    submission_dir = base_dir / "submissions"
    submission_dir.mkdir(exist_ok=True)
    
    submission = pd.read_csv(raw_data_dir / "gender_submission.csv")
    submission['Survived'] = (test_preds > 0.5).astype(int)
    submission_path = submission_dir / f"{exp_name}_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    run()
