"""
Walk-forward validation for the XAUUSD AI bot.

Splits historical data into rolling/expanding windows, trains a fresh model
on each train window, then tests on the immediately following out-of-sample
window — no data leakage between folds.

Run: python backtest/walk_forward.py

Two modes:
  - expanding : train window grows each fold (all history up to test start)
  - rolling   : fixed-size train window slides forward
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger

from data.fetcher import fetch_ohlcv, fetch_macro
from data.preprocessor import prepare_dataset
from features.pipeline import build_features, get_feature_matrix
from models.classifier import GoldDirectionClassifier
from strategy.signal import generate_signals_for_df
from backtest.engine import run_backtest, compute_metrics
from config.config import BACKTEST_START, BACKTEST_END, INITIAL_CAPITAL


# ── Walk-forward configuration ─────────────────────────────────────────────────
WF_CONFIG = {
    "mode":             "expanding",  # "expanding" or "rolling"
    "train_months":     24,           # Initial / fixed train window (months)
    "test_months":      12,           # Out-of-sample window per fold (months)
    "min_train_rows":   80,            # Skip fold if fewer training rows
    "initial_capital":  INITIAL_CAPITAL,
}


@dataclass
class FoldResult:
    fold:          int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    metrics:       dict
    equity_curve:  pd.Series
    trades:        pd.DataFrame
    feature_imp:   pd.Series
    model_auc:     float


@dataclass
class WalkForwardResult:
    folds:          list[FoldResult] = field(default_factory=list)
    combined_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    combined_equity: pd.Series    = field(default_factory=pd.Series)
    summary:        dict          = field(default_factory=dict)


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def build_fold_windows(
    start: str,
    end:   str,
    train_months: int,
    test_months:  int,
    mode: str,
) -> list[dict]:
    """Generate list of {train_start, train_end, test_start, test_end} dicts."""
    windows = []
    global_start = datetime.strptime(start, "%Y-%m-%d")
    global_end   = datetime.strptime(end,   "%Y-%m-%d")

    train_start = global_start
    train_end   = global_start + relativedelta(months=train_months)
    test_start  = train_end
    test_end    = test_start + relativedelta(months=test_months)

    fold = 1
    while test_end <= global_end:
        windows.append({
            "fold":        fold,
            "train_start": _date_str(train_start),
            "train_end":   _date_str(train_end),
            "test_start":  _date_str(test_start),
            "test_end":    _date_str(test_end),
        })

        # Advance test window
        test_start = test_end
        test_end   = test_start + relativedelta(months=test_months)

        if mode == "expanding":
            train_end = test_start   # Train window grows to include all past data
        else:
            train_start = train_start + relativedelta(months=test_months)
            train_end   = test_start

        fold += 1

    return windows


def run_fold(
    df_full: pd.DataFrame,
    window: dict,
    fold_num: int,
    initial_capital: float,
    min_train_rows: int,
) -> FoldResult | None:
    """Train on train window, test on test window. Returns None if skipped."""
    logger.info(
        f"Fold {fold_num} | Train: {window['train_start']} → {window['train_end']} "
        f"| Test: {window['test_start']} → {window['test_end']}"
    )

    train_mask = (df_full.index >= window["train_start"]) & (df_full.index < window["train_end"])
    test_mask  = (df_full.index >= window["test_start"])  & (df_full.index < window["test_end"])

    df_train = df_full[train_mask].copy()
    df_test  = df_full[test_mask].copy()

    if len(df_train) < min_train_rows:
        logger.warning(f"Fold {fold_num}: only {len(df_train)} train rows — skipping")
        return None
    if len(df_test) < 10:
        logger.warning(f"Fold {fold_num}: only {len(df_test)} test rows — skipping")
        return None

    # Train fresh model on this fold's train data
    X_train, y_train = get_feature_matrix(df_train)
    if y_train is None or len(X_train) < min_train_rows:
        logger.warning(f"Fold {fold_num}: insufficient feature rows — skipping")
        return None

    clf = GoldDirectionClassifier()
    try:
        fold_metrics = clf.train(X_train, y_train)
        model_auc = fold_metrics.get("auc", 0.0)
    except Exception as e:
        logger.error(f"Fold {fold_num} training failed: {e}")
        return None

    # Generate signals on test data only (true out-of-sample)
    X_test, _ = get_feature_matrix(df_test)
    if X_test.empty:
        return None

    probas  = clf.predict_proba(X_test)
    signals = generate_signals_for_df(df_test, probas)

    # Backtest the test period
    result = run_backtest(df_test, signals, initial_capital=initial_capital)

    logger.info(
        f"Fold {fold_num} results: WR={result.metrics.get('win_rate_pct')}% "
        f"| PF={result.metrics.get('profit_factor')} "
        f"| Return={result.metrics.get('total_return_pct')}% "
        f"| AUC={model_auc}"
    )

    return FoldResult(
        fold         = fold_num,
        train_start  = window["train_start"],
        train_end    = window["train_end"],
        test_start   = window["test_start"],
        test_end     = window["test_end"],
        metrics      = result.metrics,
        equity_curve = result.equity_curve,
        trades       = result.trades,
        feature_imp  = clf.feature_importance(),
        model_auc    = model_auc,
    )


def aggregate_results(folds: list[FoldResult], initial_capital: float) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Combine all out-of-sample trades into a single equity curve and metrics."""
    all_trades = []
    equity_pieces = []
    running_equity = initial_capital

    for fold in folds:
        if fold.trades.empty:
            continue

        # Rescale equity curve to be continuous across folds
        scale = running_equity / initial_capital
        scaled_equity = fold.equity_curve * scale
        equity_pieces.append(scaled_equity)

        # Offset trade PnL to match running equity
        trades = fold.trades.copy()
        all_trades.append(trades)

        running_equity = float(scaled_equity.iloc[-1])

    if not all_trades:
        return pd.DataFrame(), pd.Series(), {"error": "No trades across any fold"}

    combined_trades  = pd.concat(all_trades, ignore_index=True)
    combined_equity  = pd.concat(equity_pieces).sort_index()

    summary = compute_metrics(combined_trades, combined_equity, initial_capital)
    summary["n_folds"]        = len(folds)
    summary["avg_fold_auc"]   = round(np.mean([f.model_auc for f in folds]), 4)
    summary["profitable_folds"] = sum(1 for f in folds if f.metrics.get("net_profit_usd", 0) > 0)

    return combined_trades, combined_equity, summary


def plot_results(result: WalkForwardResult, save_path: str = "backtest/walk_forward_results.png") -> None:
    """Generate a comprehensive 4-panel walk-forward results chart."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("XAUUSD AI Bot — Walk-Forward Validation", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel 1: Combined out-of-sample equity curve ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(result.combined_equity, color="royalblue", linewidth=1.5, label="OOS Equity")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Starting capital")

    # Shade each fold's test period differently
    colors = plt.cm.Set3(np.linspace(0, 1, len(result.folds)))
    for fold, color in zip(result.folds, colors):
        ax1.axvspan(
            pd.Timestamp(fold.test_start),
            pd.Timestamp(fold.test_end),
            alpha=0.08, color=color, label=f"Fold {fold.fold}"
        )

    ax1.set_title("Combined Out-of-Sample Equity Curve")
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="upper left", fontsize=7, ncol=4)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Per-fold returns ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    returns = [f.metrics.get("total_return_pct", 0) for f in result.folds]
    fold_labels = [f"F{f.fold}\n{f.test_start[:7]}" for f in result.folds]
    bar_colors = ["green" if r > 0 else "red" for r in returns]
    ax2.bar(fold_labels, returns, color=bar_colors, alpha=0.75)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Return per Fold (%)")
    ax2.set_ylabel("Return %")
    ax2.grid(alpha=0.3, axis="y")

    # ── Panel 3: Per-fold win rate + AUC ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    win_rates = [f.metrics.get("win_rate_pct", 0) for f in result.folds]
    aucs      = [f.model_auc * 100 for f in result.folds]
    x = np.arange(len(result.folds))
    ax3.bar(x - 0.2, win_rates, 0.35, label="Win Rate %", color="steelblue", alpha=0.8)
    ax3.bar(x + 0.2, aucs,      0.35, label="Model AUC %", color="orange",   alpha=0.8)
    ax3.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(fold_labels)
    ax3.set_title("Win Rate vs Model AUC per Fold")
    ax3.set_ylabel("%")
    ax3.legend()
    ax3.grid(alpha=0.3, axis="y")

    # ── Panel 4: Drawdown ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if not result.combined_equity.empty:
        running_max = result.combined_equity.cummax()
        drawdown    = (result.combined_equity - running_max) / running_max * 100
        ax4.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
        ax4.plot(drawdown, color="darkred", linewidth=0.8)
    ax4.set_title("Drawdown (%)")
    ax4.set_ylabel("Drawdown %")
    ax4.grid(alpha=0.3)

    # ── Panel 5: Average feature importance across folds ─────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    all_imp = pd.concat([f.feature_imp for f in result.folds], axis=1)
    avg_imp = all_imp.mean(axis=1).sort_values(ascending=True).tail(12)
    ax5.barh(avg_imp.index, avg_imp.values, color="teal", alpha=0.75)
    ax5.set_title("Avg Feature Importance (Top 12)")
    ax5.set_xlabel("Importance")
    ax5.grid(alpha=0.3, axis="x")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Walk-forward chart saved to {save_path}")
    plt.show()


def print_summary(result: WalkForwardResult) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 65)
    print("WALK-FORWARD VALIDATION — PER FOLD RESULTS")
    print("=" * 65)
    print(f"{'Fold':<5} {'Test Period':<22} {'Return%':>8} {'WR%':>6} {'PF':>6} {'MaxDD%':>8} {'AUC':>6}")
    print("-" * 65)
    for f in result.folds:
        m = f.metrics
        period = f"{f.test_start[:7]} → {f.test_end[:7]}"
        print(
            f"{f.fold:<5} {period:<22} "
            f"{m.get('total_return_pct', 0):>7.1f}% "
            f"{m.get('win_rate_pct', 0):>5.1f}% "
            f"{m.get('profit_factor', 0):>6.2f} "
            f"{m.get('max_drawdown_pct', 0):>7.1f}% "
            f"{f.model_auc:>6.3f}"
        )
    print("=" * 65)
    print("COMBINED OUT-OF-SAMPLE RESULTS")
    print("=" * 65)
    s = result.summary
    for key, val in s.items():
        print(f"  {key:30s}: {val}")
    print("=" * 65)


def run_walk_forward(
    start:           str  = BACKTEST_START,
    end:             str  = BACKTEST_END,
    mode:            str  = WF_CONFIG["mode"],
    train_months:    int  = WF_CONFIG["train_months"],
    test_months:     int  = WF_CONFIG["test_months"],
    initial_capital: float = WF_CONFIG["initial_capital"],
    plot:            bool = True,
    save_path:       str  = "backtest/walk_forward_results.png",
) -> WalkForwardResult:
    """Full walk-forward validation pipeline."""
    logger.info(f"=== Walk-Forward Validation | mode={mode} | train={train_months}m | test={test_months}m ===")

    # 1. Fetch all data once
    ohlcv = fetch_ohlcv(start=start, end=end, interval="1D")
    try:
        macro = fetch_macro(start=start, end=end)
    except Exception as e:
        logger.warning(f"Macro data unavailable: {e}")
        macro = None

    df_full = prepare_dataset(ohlcv, macro)
    df_full = build_features(df_full, include_target=True)
    logger.info(f"Full dataset: {len(df_full)} rows ({start} → {end})")

    # 2. Build fold windows
    windows = build_fold_windows(start, end, train_months, test_months, mode)
    logger.info(f"Running {len(windows)} folds")

    # 3. Run each fold
    result = WalkForwardResult()
    for window in windows:
        fold_result = run_fold(
            df_full, window, window["fold"],
            initial_capital, WF_CONFIG["min_train_rows"]
        )
        if fold_result is not None:
            result.folds.append(fold_result)

    if not result.folds:
        logger.error("No folds completed successfully")
        return result

    # 4. Aggregate
    result.combined_trades, result.combined_equity, result.summary = \
        aggregate_results(result.folds, initial_capital)

    # 5. Report
    print_summary(result)

    # 6. Plot
    if plot:
        plot_results(result, save_path)

    return result


if __name__ == "__main__":
    run_walk_forward()
