"""
End-to-end training pipeline:
  fetch data → build features → train model → evaluate → save
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from data.fetcher import fetch_ohlcv, fetch_macro
from data.preprocessor import prepare_dataset
from features.pipeline import build_features, get_feature_matrix
from models.classifier import GoldDirectionClassifier
from config.config import BACKTEST_START, BACKTEST_END, MODEL_PATH


def run_training(
    start: str = BACKTEST_START,
    end:   str = BACKTEST_END,
    save:  bool = True,
) -> dict:
    """
    Full training run. Returns evaluation metrics.
    Uses daily bars for ML training to get maximum history (yfinance
    restricts hourly data to the last 730 days).
    """
    logger.info("=== XAUUSD AI Model Training ===")

    # 1. Fetch data — daily bars give us full history back to 2018
    ohlcv = fetch_ohlcv(start=start, end=end, interval="1D")
    try:
        macro = fetch_macro(start=start, end=end)
    except Exception as e:
        logger.warning(f"Macro data unavailable, continuing without it: {e}")
        macro = None

    # 2. Preprocess
    df = prepare_dataset(ohlcv, macro)

    # 3. Feature engineering
    df = build_features(df, include_target=True)

    # 4. Split features / target
    X, y = get_feature_matrix(df)
    logger.info(f"Feature matrix: {X.shape} | Target distribution: {y.value_counts().to_dict()}")

    # 5. Train
    clf = GoldDirectionClassifier()
    metrics = clf.train(X, y)

    # 6. Feature importance
    imp = clf.feature_importance()
    logger.info(f"Top 10 features:\n{imp.head(10)}")

    # 7. Save
    if save:
        clf.save(MODEL_PATH)

    return metrics


if __name__ == "__main__":
    metrics = run_training()
    print(f"\nTraining complete. Metrics: {metrics}")
