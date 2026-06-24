"""
Full backtest pipeline entry point.
Run: python backtest/run_backtest.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
import matplotlib.pyplot as plt

from data.fetcher import fetch_ohlcv, fetch_macro
from data.preprocessor import prepare_dataset
from features.pipeline import build_features, get_feature_matrix
from models.classifier import GoldDirectionClassifier
from strategy.signal import generate_signals_for_df
from backtest.engine import run_backtest
from config.config import BACKTEST_START, BACKTEST_END, INITIAL_CAPITAL, MODEL_PATH


def main():
    logger.info("=== XAUUSD AI Bot Backtest ===")

    # 1. Data — use daily bars to match training timeframe
    # (Yahoo Finance caps hourly to last 730 days; daily gives full history)
    ohlcv = fetch_ohlcv(start=BACKTEST_START, end=BACKTEST_END, interval="1D")
    try:
        macro = fetch_macro(start=BACKTEST_START, end=BACKTEST_END)
    except Exception as e:
        logger.warning(f"Macro data unavailable: {e}")
        macro = None

    df = prepare_dataset(ohlcv, macro)
    df = build_features(df, include_target=True)

    # 2. Load or train model
    clf = GoldDirectionClassifier()
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}")
        clf.load(MODEL_PATH)
    else:
        logger.info("No saved model found — training now")
        X, y = get_feature_matrix(df)
        clf.train(X, y)
        clf.save(MODEL_PATH)

    # 3. Generate ML probabilities
    X, _ = get_feature_matrix(df)
    probas = clf.predict_proba(X)

    # 4. Generate trade signals
    signals = generate_signals_for_df(df, probas)

    n_signals = (signals["direction"] != 0).sum()
    logger.info(f"Generated {n_signals} trade signals")
    logger.info(f"  Long:  {(signals['direction'] == 1).sum()}")
    logger.info(f"  Short: {(signals['direction'] == -1).sum()}")

    # 5. Run backtest
    result = run_backtest(df, signals, initial_capital=INITIAL_CAPITAL)

    # 6. Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, val in result.metrics.items():
        print(f"  {key:25s}: {val}")
    print("="*50)

    # 7. Plot equity curve
    if not result.equity_curve.empty:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes[0].plot(result.equity_curve, label="Equity", color="royalblue")
        axes[0].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("XAUUSD AI Bot — Equity Curve")
        axes[0].set_ylabel("Account Equity (USD)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if not result.trades.empty:
            monthly_pnl = result.trades.set_index("exit_time")["pnl_usd"].resample("ME").sum()
            colors = ["green" if v > 0 else "red" for v in monthly_pnl]
            axes[1].bar(monthly_pnl.index, monthly_pnl.values, color=colors, alpha=0.7)
            axes[1].set_title("Monthly PnL (USD)")
            axes[1].set_ylabel("PnL")
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("backtest/equity_curve.png", dpi=150)
        logger.info("Equity curve saved to backtest/equity_curve.png")
        plt.show()

    return result


if __name__ == "__main__":
    main()
