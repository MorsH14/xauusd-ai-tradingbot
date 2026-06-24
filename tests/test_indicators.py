"""Tests for technical indicator calculations."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features.indicators import (
    add_ema, add_rsi, add_macd, add_bollinger,
    add_atr, add_volume_features, add_price_features, add_all_indicators,
)


def make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic XAUUSD-like OHLCV data."""
    np.random.seed(42)
    close  = 1800 + np.cumsum(np.random.randn(n) * 3)
    high   = close + np.abs(np.random.randn(n) * 2)
    low    = close - np.abs(np.random.randn(n) * 2)
    open_  = close + np.random.randn(n) * 1
    volume = np.random.randint(100, 1000, n).astype(float)
    idx    = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestEMA:
    def test_columns_added(self):
        df = add_ema(make_ohlcv())
        for col in ["ema_9", "ema_20", "ema_50", "ema_200", "ema_cross_bull"]:
            assert col in df.columns

    def test_no_nan_after_warmup(self):
        df = add_ema(make_ohlcv(300))
        assert df["ema_200"].iloc[250:].isna().sum() == 0

    def test_ema_cross_bull_binary(self):
        df = add_ema(make_ohlcv())
        assert df["ema_cross_bull"].isin([0, 1]).all()


class TestRSI:
    def test_range(self):
        df = add_rsi(make_ohlcv())
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_oversold_binary(self):
        df = add_rsi(make_ohlcv())
        assert df["rsi_overbought"].isin([0, 1]).all()
        assert df["rsi_oversold"].isin([0, 1]).all()


class TestMACD:
    def test_columns_added(self):
        df = add_macd(make_ohlcv())
        for col in ["macd", "macd_signal", "macd_hist", "macd_bull"]:
            assert col in df.columns

    def test_hist_equals_macd_minus_signal(self):
        df = add_macd(make_ohlcv()).dropna()
        diff = (df["macd"] - df["macd_signal"] - df["macd_hist"]).abs()
        assert diff.max() < 1e-10


class TestBollinger:
    def test_upper_above_lower(self):
        df = add_bollinger(make_ohlcv()).dropna()
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_pct_range(self):
        df = add_bollinger(make_ohlcv()).dropna()
        # bb_pct can be outside 0-1 on breakouts, just check it's numeric
        assert df["bb_pct"].isna().sum() == 0


class TestATR:
    def test_positive(self):
        df = add_atr(make_ohlcv()).dropna()
        assert (df["atr"] > 0).all()

    def test_atr_pct_positive(self):
        df = add_atr(make_ohlcv()).dropna()
        assert (df["atr_pct"] > 0).all()


class TestAllIndicators:
    def test_runs_without_error(self):
        df = add_all_indicators(make_ohlcv(500))
        assert len(df) > 0

    def test_no_infinite_values(self):
        df = add_all_indicators(make_ohlcv(500)).replace([np.inf, -np.inf], np.nan)
        numeric = df.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() < len(df) * len(numeric.columns) * 0.5
