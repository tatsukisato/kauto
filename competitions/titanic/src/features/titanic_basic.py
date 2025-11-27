import pandas as pd
from pathlib import Path
from .base import BaseFeature

class TitanicBasicFeatures(BaseFeature):
    def run(self):
        train = pd.read_csv(self.input_dir / "train.csv")
        test = pd.read_csv(self.input_dir / "test.csv")
        
        # Combine for processing
        train['is_train'] = 1
        test['is_train'] = 0
        data = pd.concat([train, test], sort=False).reset_index(drop=True)
        
        # Preprocessing
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        
        # Feature Selection
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        
        # Split back
        train_processed = data[data['is_train'] == 1][features + ['Survived']].reset_index(drop=True)
        test_processed = data[data['is_train'] == 0][features].reset_index(drop=True)
        
        # Save
        self.save(train_processed.drop('Survived', axis=1), "X_train.csv")
        self.save(train_processed[['Survived']], "y_train.csv")
        self.save(test_processed, "X_test.csv")
        
        return train_processed, test_processed
