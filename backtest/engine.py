"""
Vectorized backtesting engine for XAUUSD.
Simulates trades from signal DataFrame with realistic cost modelling.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from config.config import INITIAL_CAPITAL, COMMISSION


@dataclass
class BacktestResult:
    trades:          pd.DataFrame
    equity_curve:    pd.Series
    metrics:         dict


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    commission:      float = COMMISSION,     # Per-side cost as fraction of trade value
    lot_size:        float = 0.1,            # Fixed lot for simplicity (use RiskManager for dynamic)
    contract_size:   float = 100,            # XAUUSD: 1 lot = 100 oz
) -> BacktestResult:
    """
    Vectorized backtest — iterates bar-by-bar to simulate fills.
    Signals DataFrame must have columns: direction, entry, stop_loss, take_profit
    """
    equity       = initial_capital
    peak_equity  = initial_capital
    equity_curve = []
    trades       = []

    in_trade     = False
    trade_dir    = 0
    entry_price  = 0.0
    stop_loss    = 0.0
    take_profit  = 0.0
    entry_time   = None

    for i, (ts, row) in enumerate(df.iterrows()):
        if ts not in signals.index:
            equity_curve.append({"datetime": ts, "equity": equity})
            continue

        sig = signals.loc[ts]
        close = row["close"]
        high  = row["high"]
        low   = row["low"]

        # ── Manage open trade ────────────────────────────────────────────────
        if in_trade:
            hit_stop   = (trade_dir ==  1 and low  <= stop_loss) or \
                         (trade_dir == -1 and high >= stop_loss)
            hit_target = (trade_dir ==  1 and high >= take_profit) or \
                         (trade_dir == -1 and low  <= take_profit)

            if hit_stop or hit_target:
                exit_price = take_profit if hit_target else stop_loss
                pnl_pips   = (exit_price - entry_price) * trade_dir
                pnl_usd    = pnl_pips * lot_size * contract_size
                cost       = (entry_price + exit_price) * lot_size * contract_size * commission
                net_pnl    = pnl_usd - cost

                equity     += net_pnl
                peak_equity = max(peak_equity, equity)

                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   ts,
                    "direction":   trade_dir,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "pnl_usd":     round(net_pnl, 2),
                    "winner":      net_pnl > 0,
                    "exit_reason": "TP" if hit_target else "SL",
                })

                in_trade = False

        # ── Open new trade ────────────────────────────────────────────────────
        if not in_trade and sig["direction"] != 0:
            # Skip if daily drawdown too high
            daily_dd = 1 - (equity / initial_capital)
            if daily_dd > 0.10:
                equity_curve.append({"datetime": ts, "equity": equity})
                continue

            in_trade    = True
            trade_dir   = int(sig["direction"])
            entry_price = float(sig["entry"])
            stop_loss   = float(sig["stop_loss"])
            take_profit = float(sig["take_profit"])
            entry_time  = ts

        equity_curve.append({"datetime": ts, "equity": equity})

    # Close any open trade at the end
    if in_trade:
        exit_price = df["close"].iloc[-1]
        pnl_pips   = (exit_price - entry_price) * trade_dir
        pnl_usd    = pnl_pips * lot_size * contract_size
        cost       = (entry_price + exit_price) * lot_size * contract_size * commission
        net_pnl    = pnl_usd - cost
        equity    += net_pnl
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   df.index[-1],
            "direction":   trade_dir,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "pnl_usd":     round(net_pnl, 2),
            "winner":      net_pnl > 0,
            "exit_reason": "EOD",
        })

    trades_df      = pd.DataFrame(trades)
    equity_series  = pd.DataFrame(equity_curve).set_index("datetime")["equity"]
    metrics        = compute_metrics(trades_df, equity_series, initial_capital)

    logger.info(f"Backtest complete: {metrics}")
    return BacktestResult(trades=trades_df, equity_curve=equity_series, metrics=metrics)


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    initial_capital: float,
) -> dict:
    if trades.empty:
        return {"error": "No trades taken"}

    total    = len(trades)
    winners  = trades["winner"].sum()
    win_rate = winners / total

    gross_profit = trades[trades["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_loss   = trades[trades["pnl_usd"] < 0]["pnl_usd"].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    net_profit   = trades["pnl_usd"].sum()
    total_return = net_profit / initial_capital

    # Max drawdown
    running_max  = equity.cummax()
    drawdown     = (equity - running_max) / running_max
    max_dd       = drawdown.min()

    # Sharpe ratio (annualized, assuming hourly bars)
    returns      = equity.pct_change().dropna()
    sharpe       = (returns.mean() / returns.std()) * np.sqrt(24 * 252) if returns.std() > 0 else 0

    # Average win/loss
    avg_win  = trades[trades["winner"]]["pnl_usd"].mean() if winners > 0 else 0
    avg_loss = trades[~trades["winner"]]["pnl_usd"].mean() if (total - winners) > 0 else 0

    return {
        "total_trades":     total,
        "win_rate_pct":     round(win_rate * 100, 2),
        "profit_factor":    round(profit_factor, 3),
        "net_profit_usd":   round(net_profit, 2),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "final_equity":     round(equity.iloc[-1], 2),
    }
