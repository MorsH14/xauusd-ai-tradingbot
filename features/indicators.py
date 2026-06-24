"""
Technical indicators for XAUUSD.
All functions accept a DataFrame with columns: open, high, low, close, volume
and return the same DataFrame with indicator columns appended.
"""
import pandas as pd
import numpy as np
from config.config import (
    EMA_FAST, EMA_MED, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, RSI_OB, RSI_OS,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD,
)


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df[f"ema_{EMA_FAST}"]   = df["close"].ewm(span=EMA_FAST,   adjust=False).mean()
    df[f"ema_{EMA_MED}"]    = df["close"].ewm(span=EMA_MED,    adjust=False).mean()
    df[f"ema_{EMA_SLOW}"]   = df["close"].ewm(span=EMA_SLOW,   adjust=False).mean()
    df[f"ema_{EMA_TREND}"]  = df["close"].ewm(span=EMA_TREND,  adjust=False).mean()
    # EMA crossover signal: fast > med = bullish
    df["ema_cross_bull"] = (df[f"ema_{EMA_FAST}"] > df[f"ema_{EMA_MED}"]).astype(int)
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] >= RSI_OB).astype(int)
    df["rsi_oversold"]   = (df["rsi"] <= RSI_OS).astype(int)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema_fast   = df["close"].ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow   = df["close"].ewm(span=MACD_SLOW,   adjust=False).mean()
    df["macd"]          = ema_fast - ema_slow
    df["macd_signal"]   = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]     = df["macd"] - df["macd_signal"]
    df["macd_bull"]     = (df["macd"] > df["macd_signal"]).astype(int)
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    sma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = sma + BB_STD * std
    df["bb_lower"] = sma - BB_STD * std
    df["bb_mid"]   = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    # Price position within bands (0 = lower band, 1 = upper band)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    # Normalized ATR: volatility relative to price
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_supertrend(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend trend-direction filter."""
    atr = df["atr"] if "atr" in df.columns else add_atr(df.copy())["atr"]
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        prev_upper = upper.iloc[i - 1]
        prev_lower = lower.iloc[i - 1]
        prev_close = df["close"].iloc[i - 1]

        upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < prev_upper or prev_close > prev_upper else prev_upper
        lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > prev_lower or prev_close < prev_lower else prev_lower

        if df["close"].iloc[i] > upper.iloc[i]:
            direction.iloc[i] = 1   # Bullish
        elif df["close"].iloc[i] < lower.iloc[i]:
            direction.iloc[i] = -1  # Bearish
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    df["supertrend"]      = supertrend
    df["supertrend_bull"] = (direction == 1).astype(int)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_sma20"]  = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_sma20"]
    df["high_volume"]   = (df["volume_ratio"] > 1.5).astype(int)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candle body, wick ratios, and session return."""
    df["candle_body"]    = (df["close"] - df["open"]).abs()
    df["upper_wick"]     = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]     = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"]   = df["high"] - df["low"]
    df["body_ratio"]     = df["candle_body"] / df["candle_range"].replace(0, np.nan)
    df["bullish_candle"] = (df["close"] > df["open"]).astype(int)
    # Returns
    df["return_1"]  = df["close"].pct_change(1)
    df["return_3"]  = df["close"].pct_change(3)
    df["return_5"]  = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator in order."""
    df = add_atr(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_supertrend(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    return df
