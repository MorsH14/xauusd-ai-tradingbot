"""
Cleans, resamples, and merges OHLCV with macro data.
"""
import pandas as pd
from loguru import logger


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample tick/1H bars to a higher timeframe (e.g. '4H', '1D')."""
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    logger.info(f"Resampled to {rule}: {len(resampled)} bars")
    return resampled


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info and normalize index to UTC-naive datetime."""
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = idx
    return df


def merge_macro(ohlcv: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join macro daily data onto intraday OHLCV.
    Forward-fills macro values within each day.
    Normalizes both indexes to UTC-naive to avoid dtype mismatch errors.
    """
    ohlcv = _normalize_index(ohlcv)
    macro = _normalize_index(macro)
    macro_reindexed = macro.reindex(ohlcv.index, method="ffill")
    merged = pd.concat([ohlcv, macro_reindexed], axis=1)
    merged = merged.ffill()
    missing = merged.isnull().sum()
    if missing.any():
        logger.warning(f"NaN values after macro merge:\n{missing[missing > 0]}")
    return merged


def remove_low_volume_sessions(df: pd.DataFrame, min_volume: int = 10) -> pd.DataFrame:
    """Drop bars with suspiciously low volume (e.g. market close, holidays)."""
    before = len(df)
    df = df[df["volume"] >= min_volume].copy()
    logger.info(f"Removed {before - len(df)} low-volume bars")
    return df


def filter_trading_hours(df: pd.DataFrame, start_hour: int = 7, end_hour: int = 21) -> pd.DataFrame:
    """Keep only bars within active trading hours (UTC)."""
    mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
    filtered = df[mask].copy()
    logger.info(f"Filtered to {start_hour}:00–{end_hour}:00 UTC: {len(filtered)} bars remain")
    return filtered


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check OHLCV integrity."""
    invalid = df[
        (df["high"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    ]
    if len(invalid) > 0:
        logger.warning(f"Dropping {len(invalid)} malformed OHLCV bars")
        df = df.drop(invalid.index)
    return df


def prepare_dataset(ohlcv: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = _normalize_index(ohlcv)
    df = validate_ohlcv(df)
    df = remove_low_volume_sessions(df)
    if macro is not None:
        df = merge_macro(df, macro)
    df = df.sort_index()
    return df
