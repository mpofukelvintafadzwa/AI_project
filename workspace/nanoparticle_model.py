import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class NanoparticleSizePredictor:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the RandomForest model pipeline for nanoparticle size prediction.

        Args:
            n_estimators (int): Number of trees in the random forest.
            random_state (int): Seed for reproducibility.
        """
        # Pipeline to scale features and then run RF regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
        ])
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the model on the given dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector (nanoparticle sizes).
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        """
        Predict nanoparticle sizes for input features.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted sizes.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_with_confidence(self, X):
        """
        Predict sizes with confidence intervals using the forest estimators.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: mean predictions, lower bounds, upper bounds
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        # Get predictions from each tree
        preds = np.stack([tree.predict(X) for tree in self.model.named_steps['rf'].estimators_], axis=0)
        # Mean prediction
        mean_pred = np.mean(preds, axis=0)
        # Prediction intervals: 95% confidence using percentiles
        lower = np.percentile(preds, 2.5, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        return mean_pred, lower, upper

    def save_model(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to the file where model will be saved.
        """
        if not self.is_trained:
            raise RuntimeError("Train the model before saving.")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = joblib.load(filepath)
        self.is_trained = True


# Example usage (would be removed or rewritten for production)
if __name__ == '__main__':
    # Generate synthetic dataset for demonstration
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 5 features
    y = X[:, 0]*10 + X[:, 1]*5 + np.random.normal(0, 0.5, 100)  # example size

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    predictor = NanoparticleSizePredictor(n_estimators=100)
    predictor.fit(X_train, y_train)

    # Prediction
    y_pred = predictor.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.3f}")

    # Prediction with confidence intervals
    mean_pred, lower, upper = predictor.predict_with_confidence(X_test)
    for i in range(5):
        print(f"Predicted size: {mean_pred[i]:.2f}, 95% CI: [{lower[i]:.2f}, {upper[i]:.2f}]")
