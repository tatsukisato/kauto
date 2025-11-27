import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import joblib

class LGBMModel:
    def __init__(self, params: dict, output_dir: Path):
        self.params = params
        self.output_dir = output_dir
        self.models = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X: pd.DataFrame, y: np.ndarray, cv_splits=5):
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            score = accuracy_score(y_val, val_preds)
            scores.append(score)
            print(f"Fold {fold+1} Accuracy: {score:.4f}")
            
            # Save model
            joblib.dump(model, self.output_dir / f"lgbm_fold{fold+1}.pkl")
            self.models.append(model)

        print(f"Mean Accuracy: {np.mean(scores):.4f}")
        return oof_preds, scores

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        test_preds = np.zeros(len(X_test))
        for model in self.models:
            test_preds += model.predict_proba(X_test)[:, 1] / len(self.models)
        return test_preds
