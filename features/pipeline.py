"""
Master feature pipeline: runs all feature builders and returns
a clean DataFrame ready for model training or live inference.
"""
import pandas as pd
import numpy as np
from loguru import logger
from features.indicators import add_all_indicators
from features.smc import add_smc_features
from features.macro import add_all_macro_features
from config.config import PREDICTION_HORIZON


def build_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Binary classification target:
      1 = price is higher N bars from now (buy signal)
      0 = price is lower or flat N bars from now (no trade / sell)
    """
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype(int)
    return df


def build_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    logger.info("Building features...")

    df = add_all_indicators(df)
    df = add_smc_features(df)
    df = add_all_macro_features(df)

    if include_target:
        df = build_target(df)

    # Drop rows with NaN from indicator warm-up
    before = len(df)
    df = df.dropna()
    logger.info(f"Feature pipeline complete: {len(df)}/{before} rows retained")
    return df


FEATURE_COLUMNS = [
    # EMA
    "ema_9", "ema_20", "ema_50", "ema_200", "ema_cross_bull",
    # RSI
    "rsi", "rsi_overbought", "rsi_oversold",
    # MACD
    "macd", "macd_signal", "macd_hist", "macd_bull",
    # Bollinger
    "bb_width", "bb_pct",
    # ATR / volatility
    "atr", "atr_pct",
    # Supertrend
    "supertrend_bull",
    # Volume
    "volume_ratio", "high_volume",
    # Price features
    "candle_body", "upper_wick", "lower_wick", "body_ratio",
    "bullish_candle", "return_1", "return_3", "return_5", "return_10",
    # SMC
    "swing_high", "swing_low",
    "bullish_ob", "bearish_ob",
    "bullish_fvg", "bearish_fvg", "fvg_size",
    "bos_bull", "bos_bear",
    # Session
    "london_session", "ny_session", "overlap_session",
    "asian_session", "pm_fix_window", "avoid_session",
    "hour", "day_of_week",
    # Macro (only present if macro data was merged)
    "dxy_trend", "dxy_return", "yield_rising", "macro_bull_gold", "macro_bear_gold",
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y) where X has only available feature columns."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df["target"] if "target" in df.columns else None
    return X, y
