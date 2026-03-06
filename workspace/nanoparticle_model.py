"""
Nanoparticle Size Prediction Module

This module provides functionality to train and use a RandomForest regression model
for predicting nanoparticle sizes based on synthesis parameters and other features.

Features:
- Model training with hyperparameter options
- Prediction interface
- Model saving and loading
- Basic explainability using feature importances

Dependencies:
- scikit-learn
- pandas
- joblib

"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load


class NanoparticleSizeModel:
    def __init__(self, model_path=None, n_estimators=100, random_state=42):
        """
        Initialize the nanoparticle size prediction model.

        :param model_path: Path to a pre-trained model to load. If None, a new model is created.
        :param n_estimators: Number of trees in the Random Forest.
        :param random_state: Random seed.
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.random_state = random_state

        if model_path and os.path.isfile(model_path):
            self.model = load(model_path)
            self.is_trained = True
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
            self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42, save_model=True):
        """
        Train the RandomForest regression model.

        :param X: Features dataframe
        :param y: Target series (nanoparticle sizes)
        :param test_size: Fraction of data for validation
        :param random_state: Seed for train-test splitting
        :param save_model: Whether to save the model to disk after training
        :return: Dictionary with train and validation RMSE and R2
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        if save_model and self.model_path:
            self.save_model(self.model_path)

        return {
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_r2": train_r2,
            "val_r2": val_r2
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict nanoparticle sizes given input features.

        :param X: Features dataframe
        :return: Predictions as numpy array
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before prediction.")
        preds = self.model.predict(X)
        return preds

    def save_model(self, path: str):
        """
        Save the trained model to a file.

        :param path: File path to save the model
        """
        dump(self.model, path)

    def load_model(self, path: str):
        """
        Load a trained model from a file.

        :param path: File path to load the model from
        """
        self.model = load(path)
        self.is_trained = True
        self.model_path = path

    def feature_importances(self, feature_names=None) -> pd.DataFrame:
        """
        Retrieve feature importances from the trained model.

        :param feature_names: List of feature names corresponding to the model input features
        :return: DataFrame sorted by importance descending
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded to get feature importances.")

        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


# Example usage (to be removed or replaced with unit tests in production)
if __name__ == "__main__":
    # Generate dummy data for demonstration
    np.random.seed(42)
    n_samples = 500
    feature_count = 5
    X_dummy = pd.DataFrame(np.random.rand(n_samples, feature_count), columns=[f"param_{i}" for i in range(feature_count)])
    # Simulate nanoparticle size as a function of features + noise
    y_dummy = (X_dummy["param_0"] * 50 + X_dummy["param_1"] * 30 + np.random.randn(n_samples) * 2)

    model = NanoparticleSizeModel(model_path="nanoparticle_rf_model.joblib")
    metrics = model.train(X_dummy, y_dummy)
    print(f"Training completed. Validation RMSE: {metrics['val_rmse']:.3f}, R2: {metrics['val_r2']:.3f}")

    feat_imp = model.feature_importances(X_dummy.columns.tolist())
    print("Feature Importances:")
    print(feat_imp)

    # Predict on new data (here reuse part of training data)
    preds = model.predict(X_dummy.head(5))
    print("Sample Predictions:")
    print(preds)
