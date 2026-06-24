"""
Smart Money Concepts (SMC) features:
- Order Blocks (OB): institutional supply/demand zones
- Fair Value Gaps (FVG): price imbalances institutions tend to fill
- Break of Structure (BOS): trend shift confirmation
- Swing highs/lows: key structural levels
"""
import pandas as pd
import numpy as np
from config.config import OB_LOOKBACK, FVG_MIN_SIZE, SWING_LOOKBACK


def find_swing_highs_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Mark swing highs and lows.
    A swing high is a bar whose high is the highest in the surrounding window.
    """
    highs = df["high"]
    lows  = df["low"]

    df["swing_high"] = 0
    df["swing_low"]  = 0

    for i in range(lookback, len(df) - lookback):
        window_highs = highs.iloc[i - lookback: i + lookback + 1]
        window_lows  = lows.iloc[i - lookback: i + lookback + 1]

        if highs.iloc[i] == window_highs.max():
            df.at[df.index[i], "swing_high"] = 1
        if lows.iloc[i] == window_lows.min():
            df.at[df.index[i], "swing_low"] = 1

    return df


def find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> pd.DataFrame:
    """
    Identify bullish and bearish order blocks.
    Bullish OB: last down-candle before a strong up-move (institutional buy zone).
    Bearish OB: last up-candle before a strong down-move (institutional sell zone).
    """
    df["bullish_ob"] = 0
    df["bearish_ob"] = 0
    df["ob_high"]    = np.nan
    df["ob_low"]     = np.nan

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values

    for i in range(lookback + 2, len(df)):
        # Check for bullish OB: strong bullish engulfing after a series of sells
        if close[i] > open_[i]:  # current candle is bullish
            # Find last bearish candle in lookback
            for j in range(i - 1, max(i - lookback, 0), -1):
                if close[j] < open_[j]:  # bearish candle = potential OB
                    # Confirm: candle i broke above bearish OB high
                    if close[i] > high[j]:
                        df.at[df.index[j], "bullish_ob"] = 1
                        df.at[df.index[j], "ob_high"]    = high[j]
                        df.at[df.index[j], "ob_low"]     = low[j]
                    break

        # Check for bearish OB: strong bearish engulfing after a series of buys
        if close[i] < open_[i]:  # current candle is bearish
            for j in range(i - 1, max(i - lookback, 0), -1):
                if close[j] > open_[j]:  # bullish candle = potential OB
                    if close[i] < low[j]:
                        df.at[df.index[j], "bearish_ob"] = 1
                        df.at[df.index[j], "ob_high"]    = high[j]
                        df.at[df.index[j], "ob_low"]     = low[j]
                    break

    return df


def find_fair_value_gaps(df: pd.DataFrame, min_size: float = FVG_MIN_SIZE) -> pd.DataFrame:
    """
    Fair Value Gaps (FVG) / Imbalances:
    Bullish FVG: gap between candle[i-2].high and candle[i].low (price moved up too fast).
    Bearish FVG: gap between candle[i-2].low and candle[i].high.
    """
    df["bullish_fvg"] = 0
    df["bearish_fvg"] = 0
    df["fvg_size"]    = 0.0

    for i in range(2, len(df)):
        # Bullish FVG: gap above candle i-2
        bull_gap = df["low"].iloc[i] - df["high"].iloc[i - 2]
        if bull_gap > min_size:
            df.at[df.index[i], "bullish_fvg"] = 1
            df.at[df.index[i], "fvg_size"]    = bull_gap

        # Bearish FVG: gap below candle i-2
        bear_gap = df["low"].iloc[i - 2] - df["high"].iloc[i]
        if bear_gap > min_size:
            df.at[df.index[i], "bearish_fvg"] = 1
            df.at[df.index[i], "fvg_size"]    = bear_gap

    return df


def find_break_of_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break of Structure (BOS):
    Bullish BOS: price closes above the last swing high → trend turning up.
    Bearish BOS: price closes below the last swing low → trend turning down.
    """
    df["bos_bull"] = 0
    df["bos_bear"] = 0

    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        df = find_swing_highs_lows(df)

    last_swing_high = np.nan
    last_swing_low  = np.nan

    for i in range(len(df)):
        row = df.iloc[i]

        if row["swing_high"] == 1:
            last_swing_high = row["high"]
        if row["swing_low"] == 1:
            last_swing_low = row["low"]

        if not np.isnan(last_swing_high) and row["close"] > last_swing_high:
            df.at[df.index[i], "bos_bull"] = 1
        if not np.isnan(last_swing_low) and row["close"] < last_swing_low:
            df.at[df.index[i], "bos_bear"] = 1

    return df


def add_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all SMC feature detectors."""
    df = find_swing_highs_lows(df)
    df = find_order_blocks(df)
    df = find_fair_value_gaps(df)
    df = find_break_of_structure(df)
    return df
