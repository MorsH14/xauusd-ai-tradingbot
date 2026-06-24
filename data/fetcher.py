"""
Fetches OHLCV data for XAUUSD from yfinance (research/backtest)
and MetaTrader 5 (live trading).
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from config.config import (
    SYMBOL_YFINANCE, DXY_TICKER, TIPS_10Y_TICKER, FED_FUNDS_TICKER,
    BACKTEST_START, BACKTEST_END
)

INTERVAL_MAP = {
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1H": "1h", "4H": "1h",   # yfinance has no 4H; resample in preprocessor
    "1D": "1d", "1W": "1wk",
}

# Yahoo Finance limits: intraday data only available for the last N days
INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "1h": 730,
}


def _clamp_start_for_interval(yf_interval: str, start: str) -> str:
    """Ensure start date is within Yahoo Finance's allowed range for the interval."""
    max_days = INTRADAY_MAX_DAYS.get(yf_interval)
    if max_days is None:
        return start  # Daily/weekly — no restriction
    earliest = datetime.utcnow() - timedelta(days=max_days - 1)
    requested = datetime.strptime(start, "%Y-%m-%d")
    if requested < earliest:
        clamped = earliest.strftime("%Y-%m-%d")
        logger.warning(
            f"Yahoo Finance only provides {yf_interval} data for the last {max_days} days. "
            f"Clamping start from {start} → {clamped}"
        )
        return clamped
    return start


def fetch_ohlcv(
    ticker: str = SYMBOL_YFINANCE,
    interval: str = "1H",
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
) -> pd.DataFrame:
    """Download OHLCV bars from Yahoo Finance."""
    yf_interval = INTERVAL_MAP.get(interval, interval)
    start = _clamp_start_for_interval(yf_interval, start)
    logger.info(f"Fetching {ticker} | {yf_interval} | {start} → {end}")

    df = yf.download(ticker, start=start, end=end, interval=yf_interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({yf_interval})")

    # yfinance may return MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"
    df = df.dropna()

    logger.info(f"Fetched {len(df)} bars")
    return df


def fetch_macro(start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """
    Fetch macro features on daily timeframe:
    - DXY (US Dollar Index)
    - 10Y Treasury yield
    - Fed Funds Rate proxy (13-week T-Bill)
    """
    logger.info("Fetching macro data (DXY, TIPS, FedFunds)")

    tickers = {
        "dxy":       DXY_TICKER,
        "yield_10y": TIPS_10Y_TICKER,
        "fed_funds":  FED_FUNDS_TICKER,
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end, interval="1d",
                               auto_adjust=True, progress=False)
            if not data.empty:
                close = data["Close"]
                # Flatten MultiIndex if yfinance returns one
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                frames[name] = close.rename(name)
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")

    if not frames:
        raise RuntimeError("Failed to fetch any macro data")

    macro = pd.concat(frames.values(), axis=1)
    macro.index.name = "datetime"
    macro = macro.ffill().dropna()
    logger.info(f"Macro data: {len(macro)} daily rows")
    return macro


def fetch_mt5_ohlcv(symbol: str, timeframe, n_bars: int = 5000) -> pd.DataFrame:
    """
    Fetch OHLCV from MetaTrader 5 (live use only).
    Requires MT5 terminal to be running and connected.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 package not installed. Run: pip install MetaTrader5")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data for {symbol}")

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "tick_volume"]].copy()
    df = df.rename(columns={"tick_volume": "volume"})
    return df
