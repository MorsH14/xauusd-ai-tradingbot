"""
Signal generation engine for XAUUSD.
Combines ML model probability with technical confluence rules
to produce high-confidence trade signals.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
from strategy.regime import Regime, detect_regime, regime_allows_trade
from config.config import ATR_STOP_MULTIPLIER, RISK_PER_TRADE


@dataclass
class TradeSignal:
    direction:   int        # 1 = long, -1 = short, 0 = no trade
    entry_price: float
    stop_loss:   float
    take_profit: float
    confidence:  float      # ML model probability [0, 1]
    regime:      str
    reason:      str        # Human-readable explanation


NO_TRADE = TradeSignal(0, 0.0, 0.0, 0.0, 0.0, "unknown", "No signal")


def _check_technical_confluence(row: pd.Series, direction: int) -> tuple[bool, str]:
    """
    Require multiple technical conditions to align before entering.
    Returns (passes, reason_string).
    """
    reasons = []

    if direction == 1:  # LONG
        checks = {
            "EMA trend bullish":    row.get("ema_cross_bull", 0) == 1,
            "Price > EMA200":       row.get("close", 0) > row.get("ema_200", 0),
            "RSI not overbought":   row.get("rsi_overbought", 0) == 0,
            "MACD bullish":         row.get("macd_bull", 0) == 1,
            "Supertrend bullish":   row.get("supertrend_bull", 0) == 1,
        }
    else:               # SHORT
        checks = {
            "EMA trend bearish":    row.get("ema_cross_bull", 1) == 0,
            "Price < EMA200":       row.get("close", 0) < row.get("ema_200", 0),
            "RSI not oversold":     row.get("rsi_oversold", 0) == 0,
            "MACD bearish":         row.get("macd_bull", 1) == 0,
            "Supertrend bearish":   row.get("supertrend_bull", 1) == 0,
        }

    passed = sum(checks.values())
    total  = len(checks)

    for name, result in checks.items():
        if result:
            reasons.append(f"✓ {name}")

    # Require at least 3 out of 5 conditions
    return passed >= 3, f"{passed}/{total} conditions: " + ", ".join(reasons)


def _check_smc_confluence(row: pd.Series, direction: int) -> tuple[bool, str]:
    """SMC-layer confluence check (order blocks, FVGs, BOS)."""
    if direction == 1:
        bos    = row.get("bos_bull", 0) == 1
        ob     = row.get("bullish_ob", 0) == 1
        fvg    = row.get("bullish_fvg", 0) == 1
        macro  = row.get("macro_bull_gold", 0) == 1
        smc_ok = bos or ob or fvg
        reason = f"BOS_bull={bos}, OB_bull={ob}, FVG_bull={fvg}, Macro_bull={macro}"
    else:
        bos    = row.get("bos_bear", 0) == 1
        ob     = row.get("bearish_ob", 0) == 1
        fvg    = row.get("bearish_fvg", 0) == 1
        macro  = row.get("macro_bear_gold", 0) == 1
        smc_ok = bos or ob or fvg
        reason = f"BOS_bear={bos}, OB_bear={ob}, FVG_bear={fvg}, Macro_bear={macro}"

    return smc_ok, reason


def generate_signal(
    row: pd.Series,
    ml_proba: float,
    regime: Regime,
    long_threshold:  float = 0.58,
    short_threshold: float = 0.42,
    rr_ratio:        float = 2.0,
) -> TradeSignal:
    """
    Generate a trade signal for a single bar.
    - ml_proba: probability of upward move from the ML model
    - regime:   current market regime
    """
    atr   = row.get("atr", 1.0)
    close = row.get("close", 0.0)
    avoid = row.get("avoid_session", 0) == 1

    if avoid:
        return TradeSignal(0, close, 0, 0, ml_proba, regime.value, "Avoid session (Friday close / Monday open)")

    # Determine direction from ML probability
    if ml_proba >= long_threshold:
        direction = 1
    elif ml_proba <= short_threshold:
        direction = -1
    else:
        return TradeSignal(0, close, 0, 0, ml_proba, regime.value, f"ML probability {ml_proba:.2f} in neutral zone")

    # Check regime suitability
    if not regime_allows_trade(regime, strategy_type="trend"):
        return TradeSignal(0, close, 0, 0, ml_proba, regime.value, f"Regime {regime.value} not suitable for trend trade")

    # Technical confluence gate
    tech_ok, tech_reason = _check_technical_confluence(row, direction)
    smc_ok,  smc_reason  = _check_smc_confluence(row, direction)

    if not (tech_ok or smc_ok):
        return TradeSignal(0, close, 0, 0, ml_proba, regime.value,
                           f"Confluence failed. Tech: {tech_reason} | SMC: {smc_reason}")

    # Calculate entry, stop, target
    stop_distance = ATR_STOP_MULTIPLIER * atr
    if direction == 1:
        stop_loss   = close - stop_distance
        take_profit = close + stop_distance * rr_ratio
    else:
        stop_loss   = close + stop_distance
        take_profit = close - stop_distance * rr_ratio

    reason = (f"ML={ml_proba:.2f} | {tech_reason} | SMC: {smc_reason}")
    return TradeSignal(direction, close, stop_loss, take_profit, ml_proba, regime.value, reason)


def generate_signals_for_df(
    df: pd.DataFrame,
    ml_probas: np.ndarray,
) -> pd.DataFrame:
    """Apply signal generation across an entire DataFrame (for backtesting)."""
    regimes = detect_regime(df)

    signals = []
    for i, (idx, row) in enumerate(df.iterrows()):
        sig = generate_signal(row, ml_probas[i], regimes.iloc[i])
        signals.append({
            "datetime":   idx,
            "direction":  sig.direction,
            "entry":      sig.entry_price,
            "stop_loss":  sig.stop_loss,
            "take_profit": sig.take_profit,
            "confidence": sig.confidence,
            "regime":     sig.regime,
            "reason":     sig.reason,
        })

    return pd.DataFrame(signals).set_index("datetime")
