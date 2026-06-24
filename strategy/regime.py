"""
Market regime detection for XAUUSD.
Identifies whether the market is trending, ranging, or volatile —
so the correct strategy is applied for each condition.
"""
import pandas as pd
import numpy as np
from enum import Enum


class Regime(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING       = "ranging"
    HIGH_VOL      = "high_vol"      # News/event-driven spike — avoid trading
    UNKNOWN       = "unknown"


def detect_regime(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Classify each bar's market regime using:
    - ADX-like trend strength (EMA spread)
    - ATR percentile (volatility level)
    - Price vs EMA-200 (macro trend)
    """
    regimes = pd.Series(Regime.UNKNOWN, index=df.index)

    if "atr" not in df.columns or "ema_200" not in df.columns:
        return regimes

    atr_pct     = df["atr"] / df["close"]
    atr_high    = atr_pct > atr_pct.rolling(lookback * 5).quantile(0.85)

    above_ema200 = df["close"] > df["ema_200"]
    ema_spread   = (df["ema_9"] - df["ema_50"]) / df["close"]
    trending     = ema_spread.abs() > 0.003   # EMAs separated by >0.3% of price

    for i in range(len(df)):
        if atr_high.iloc[i]:
            regimes.iloc[i] = Regime.HIGH_VOL
        elif trending.iloc[i]:
            if above_ema200.iloc[i] and ema_spread.iloc[i] > 0:
                regimes.iloc[i] = Regime.TRENDING_BULL
            else:
                regimes.iloc[i] = Regime.TRENDING_BEAR
        else:
            regimes.iloc[i] = Regime.RANGING

    return regimes


def regime_allows_trade(regime: Regime, strategy_type: str = "trend") -> bool:
    """
    Decide if current regime is suitable for the given strategy type.
    - "trend"     → trades in TRENDING_BULL / TRENDING_BEAR
    - "reversion" → trades in RANGING
    - "breakout"  → trades in all except HIGH_VOL
    """
    if regime == Regime.HIGH_VOL:
        return False
    if strategy_type == "trend":
        return regime in (Regime.TRENDING_BULL, Regime.TRENDING_BEAR)
    if strategy_type == "reversion":
        return regime == Regime.RANGING
    if strategy_type == "breakout":
        return True
    return False
