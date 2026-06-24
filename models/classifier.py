"""
XGBoost classifier for XAUUSD directional prediction.
Predicts whether price will be higher N bars in the future (binary: 1/0).
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from loguru import logger
from config.config import MODEL_PATH, TRAIN_SPLIT


# Hyperparameters tuned for financial time-series (conservative, low overfitting)
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,   # High value prevents fitting noise
    "gamma":             1.0,
    "reg_alpha":         0.5,
    "reg_lambda":        1.0,
    "scale_pos_weight":  1.0,  # Adjust if class imbalance detected
    "eval_metric":       "auc",
    "use_label_encoder": False,
    "random_state":      42,
    "n_jobs":            -1,
}


class GoldDirectionClassifier:
    def __init__(self, params: dict = XGB_PARAMS):
        self.model  = XGBClassifier(**params)
        self.params = params
        self.feature_names: list[str] = []
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train with time-series cross-validation (no look-ahead leakage).
        Returns performance metrics on the out-of-sample fold.
        """
        self.feature_names = list(X.columns)

        # Strict chronological train/test split — never shuffle financial data
        split_idx = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training on {len(X_train)} samples, evaluating on {len(X_test)}")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds      = self.model.predict(X_test)
        proba      = self.model.predict_proba(X_test)[:, 1]
        auc        = roc_auc_score(y_test, proba)
        precision  = precision_score(y_test, preds, zero_division=0)

        metrics = {
            "auc":       round(auc, 4),
            "precision": round(precision, 4),
            "n_train":   len(X_train),
            "n_test":    len(X_test),
        }
        logger.info(f"Model metrics: {metrics}")
        logger.info(f"\n{classification_report(y_test, preds)}")

        self.is_trained = True
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of upward move for each bar."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        available = [c for c in self.feature_names if c in X.columns]
        return self.model.predict_proba(X[available])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.55) -> np.ndarray:
        """Return binary signal: 1 = buy, 0 = no trade."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)
        return imp

    def save(self, path: str = MODEL_PATH) -> None:
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = MODEL_PATH) -> None:
        obj = joblib.load(path)
        self.model         = obj["model"]
        self.feature_names = obj["features"]
        self.is_trained    = True
        logger.info(f"Model loaded from {path}")
