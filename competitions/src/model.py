"""
Model module for atmaCup #22
"""
import lightgbm as lgb
import numpy as np
from pathlib import Path
from typing import Optional


class LGBMClassifier:
    """LightGBM classifier for player identification"""
    
    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ):
        """
        Args:
            params: LightGBM parameters
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
        """
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        
        # Default parameters
        default_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        if params is not None:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Create label mapping (original label -> continuous index)
        unique_labels = np.unique(y_train)
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.inverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Convert labels to continuous indices
        y_train_mapped = np.array([self.label_mapping[label] for label in y_train])
        if y_val is not None:
            y_val_mapped = np.array([self.label_mapping.get(label, 0) for label in y_val])
        
        # Get number of classes
        num_classes = len(unique_labels)
        self.params['num_class'] = num_classes
        
        print(f"Training LightGBM with {num_classes} classes")
        print(f"Label mapping: {self.label_mapping}")
        print(f"Training samples: {len(X_train)}")
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train_mapped)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
            print(f"Validation samples: {len(X_val)}")
        
        # Train
        callbacks = [
            lgb.log_evaluation(period=100),
        ]
        
        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            )
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        print(f"Training completed. Best iteration: {self.model.best_iteration}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels (original labels)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get probabilities
        probs = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Get class with highest probability (continuous index)
        predictions_idx = np.argmax(probs, axis=1)
        
        # Convert back to original labels
        predictions = np.array([self.inverse_mapping[idx] for idx in predictions_idx])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        probs = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return probs
    
    def save_model(self, path: str):
        """Save model to file
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file
        
        Args:
            path: Path to load model from
        """
        self.model = lgb.Booster(model_file=path)
        print(f"Model loaded from {path}")
