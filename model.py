"""
Personal Finance Coach - ML Model
Uses linear regression to predict future savings based on spending patterns.
"""

import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

MODEL_PATH = "savings_model.joblib"


class SavingsPredictor:
    """ML model to predict monthly savings based on expense patterns."""

    def __init__(self):
        self.model = None
        self.feature_names = ["income", "food", "rent", "travel", "shopping",
                               "utilities", "entertainment", "healthcare", "other"]
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train a new one if none exists."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                return
            except Exception:
                pass

        # Train new model if no saved model exists
        self._train_model()

    def _prepare_features(self, records: list) -> tuple:
        """Extract features (X) and target (y) from records."""
        if len(records) < 3:
            return None, None

        X = []
        y = []

        for r in records:
            features = [r.get(f, 0) for f in self.feature_names]
            X.append(features)
            y.append(r.get("savings", 0))

        return np.array(X), np.array(y)

    def _train_model(self):
        """Train the savings prediction model."""
        from main import FinanceData

        records = FinanceData.get_records_asFloats()
        if len(records) < 3:
            self.model = None
            return

        X, y = self._prepare_features(records)
        if X is None or len(X) < 3:
            self.model = None
            return

        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Use Ridge regression for better generalization
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel trained - MAE: ${mae:.2f}, R²: {r2:.3f}")

        # Save model
        joblib.dump(self.model, MODEL_PATH)

    def retrain(self):
        """Force retrain the model with current data."""
        self._train_model()

    def predict(self, features: np.ndarray) -> float:
        """Predict savings for given expense features.

        Args:
            features: Array of [income, food, rent, travel, shopping,
                               utilities, entertainment, healthcare, other]

        Returns:
            Predicted savings amount
        """
        if self.model is None:
            # Fallback: simple calculation
            total_expense = sum(features[1:])  # All except income
            return features[0] - total_expense

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        return self.model.predict(features)[0]

    def predict_from_expenses(self, income: float, expenses: dict) -> float:
        """Predict savings from income and expense dictionary."""
        features = [income]
        for f in self.feature_names[1:]:  # Skip income (first element)
            features.append(expenses.get(f, 0))
        return self.predict(np.array(features))

    def get_feature_importance(self) -> dict:
        """Return feature importance scores (coefficients)."""
        if self.model is None:
            return {}

        importance = {}
        for name, coef in zip(self.feature_names, self.model.coef_):
            importance[name] = float(coef)
        return importance

    def evaluate(self) -> dict:
        """Evaluate model performance on training data."""
        from main import FinanceData

        records = FinanceData.get_records_asFloats()
        if len(records) < 3:
            return {"error": "Insufficient data for evaluation"}

        X, y = self._prepare_features(records)
        if X is None:
            return {"error": "Could not prepare features"}

        y_pred = self.model.predict(X)

        return {
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
            "sample_count": len(y)
        }
