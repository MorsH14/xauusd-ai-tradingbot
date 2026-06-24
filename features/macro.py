"""
Macro regime features derived from DXY, yields, and Fed policy.
These are the fundamental drivers of gold price direction.
"""
import pandas as pd
import numpy as np


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived macro signals from raw macro columns.
    Expects columns: dxy, yield_10y, fed_funds (merged from macro data).
    """
    if "dxy" in df.columns:
        df["dxy_trend"]  = (df["dxy"] > df["dxy"].ewm(span=20).mean()).astype(int)
        df["dxy_return"] = df["dxy"].pct_change(5)

    if "yield_10y" in df.columns:
        df["yield_rising"] = (df["yield_10y"].diff(5) > 0).astype(int)
        df["real_yield"]   = df["yield_10y"] - (df.get("cpi_proxy", 0))

    if "fed_funds" in df.columns:
        df["rate_trend"] = (df["fed_funds"].diff(20) > 0).astype(int)

    # Macro regime: bullish for gold when DXY weak + yields falling
    if "dxy" in df.columns and "yield_10y" in df.columns:
        dxy_weak      = df["dxy"] < df["dxy"].ewm(span=20).mean()
        yields_falling = df["yield_10y"].diff(5) < 0
        df["macro_bull_gold"] = (dxy_weak & yields_falling).astype(int)
        df["macro_bear_gold"] = (~dxy_weak & ~yields_falling).astype(int)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day and day-of-week features (UTC index expected)."""
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek   # 0=Mon, 4=Fri

    # Session flags
    df["london_session"]   = ((df["hour"] >= 7)  & (df["hour"] < 16)).astype(int)
    df["ny_session"]       = ((df["hour"] >= 12) & (df["hour"] < 21)).astype(int)
    df["overlap_session"]  = ((df["hour"] >= 13) & (df["hour"] < 17)).astype(int)
    df["asian_session"]    = ((df["hour"] >= 0)  & (df["hour"] < 7)).astype(int)

    # London PM fix (10:00 EST = 15:00 UTC) — high institutional activity
    df["pm_fix_window"] = ((df["hour"] >= 14) & (df["hour"] <= 16)).astype(int)

    # Avoid trading Friday close (illiquid) and Monday open (gap risk)
    df["avoid_session"] = ((df["day_of_week"] == 4) & (df["hour"] >= 20)).astype(int)
    df["avoid_session"] |= ((df["day_of_week"] == 0) & (df["hour"] < 8)).astype(int)

    return df


def add_all_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_macro_features(df)
    df = add_session_features(df)
    return df
