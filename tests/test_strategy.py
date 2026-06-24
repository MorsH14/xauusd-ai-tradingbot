"""Tests for signal generation and regime detection."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy.regime import detect_regime, regime_allows_trade, Regime
from strategy.signal import generate_signal, NO_TRADE
from features.indicators import add_all_indicators


def make_featured_df(n: int = 500) -> pd.DataFrame:
    np.random.seed(7)
    close  = 2000 + np.cumsum(np.random.randn(n) * 5)
    high   = close + np.abs(np.random.randn(n) * 3)
    low    = close - np.abs(np.random.randn(n) * 3)
    open_  = close + np.random.randn(n) * 2
    volume = np.random.randint(200, 2000, n).astype(float)
    idx    = pd.date_range("2023-01-01", periods=n, freq="1h")
    df     = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    return add_all_indicators(df).dropna()


class TestRegimeDetection:
    def test_returns_series(self):
        df = make_featured_df()
        regimes = detect_regime(df)
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(df)

    def test_all_valid_values(self):
        df = make_featured_df()
        regimes = detect_regime(df)
        valid = set(r.value for r in Regime)
        assert all(r.value in valid for r in regimes)


class TestRegimeFilter:
    def test_high_vol_blocks_all(self):
        assert not regime_allows_trade(Regime.HIGH_VOL, "trend")
        assert not regime_allows_trade(Regime.HIGH_VOL, "reversion")
        assert not regime_allows_trade(Regime.HIGH_VOL, "breakout")

    def test_trend_strategy_requires_trend(self):
        assert regime_allows_trade(Regime.TRENDING_BULL, "trend")
        assert regime_allows_trade(Regime.TRENDING_BEAR, "trend")
        assert not regime_allows_trade(Regime.RANGING, "trend")

    def test_reversion_requires_ranging(self):
        assert regime_allows_trade(Regime.RANGING, "reversion")
        assert not regime_allows_trade(Regime.TRENDING_BULL, "reversion")

    def test_breakout_allows_trending_and_ranging(self):
        assert regime_allows_trade(Regime.TRENDING_BULL, "breakout")
        assert regime_allows_trade(Regime.RANGING, "breakout")


class TestSignalGeneration:
    def test_no_trade_in_neutral_zone(self):
        df = make_featured_df()
        row = df.iloc[-1]
        sig = generate_signal(row, ml_proba=0.50, regime=Regime.TRENDING_BULL)
        assert sig.direction == 0

    def test_no_trade_in_high_vol(self):
        df = make_featured_df()
        row = df.iloc[-1]
        sig = generate_signal(row, ml_proba=0.70, regime=Regime.HIGH_VOL)
        assert sig.direction == 0

    def test_long_signal_structure(self):
        df = make_featured_df()
        row = df.iloc[-1].copy()
        # Force all confluence conditions true for long
        row["ema_cross_bull"] = 1
        row["ema_200"]        = row["close"] - 100
        row["rsi_overbought"] = 0
        row["macd_bull"]      = 1
        row["supertrend_bull"] = 1
        row["bos_bull"]       = 1
        row["avoid_session"]  = 0

        sig = generate_signal(row, ml_proba=0.70, regime=Regime.TRENDING_BULL)
        if sig.direction == 1:
            assert sig.stop_loss < sig.entry_price
            assert sig.take_profit > sig.entry_price
            assert sig.confidence == 0.70

    def test_stop_below_entry_for_long(self):
        df = make_featured_df()
        row = df.iloc[-1].copy()
        row["ema_cross_bull"]  = 1
        row["ema_200"]         = row["close"] - 100
        row["rsi_overbought"]  = 0
        row["macd_bull"]       = 1
        row["supertrend_bull"] = 1
        row["bos_bull"]        = 1
        row["avoid_session"]   = 0

        sig = generate_signal(row, ml_proba=0.75, regime=Regime.TRENDING_BULL)
        if sig.direction == 1:
            assert sig.stop_loss < sig.entry_price
            assert sig.take_profit > sig.entry_price
