import os
from dotenv import load_dotenv

load_dotenv()

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL = "XAUUSD"
SYMBOL_YFINANCE = "GC=F"        # Gold futures on Yahoo Finance
SYMBOL_MT5 = "XAUUSD"

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAME_BIAS   = "1D"         # Trend direction
TIMEFRAME_CONFIRM = "4H"        # Confirmation
TIMEFRAME_ENTRY  = "1H"         # Entry trigger
TIMEFRAME_EXEC   = "15min"      # Execution precision

# ── Trading sessions (UTC) ────────────────────────────────────────────────────
LONDON_OPEN_UTC  = "07:00"
LONDON_CLOSE_UTC = "16:00"
NY_OPEN_UTC      = "12:00"
NY_CLOSE_UTC     = "21:00"
# Peak window: London/NY overlap 13:00–17:00 UTC
PEAK_SESSION_START = "13:00"
PEAK_SESSION_END   = "17:00"

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE       = 0.01     # 1% of account equity per trade
MAX_DAILY_DRAWDOWN   = 0.03     # Pause trading if daily loss > 3%
MAX_TOTAL_DRAWDOWN   = 0.10     # Hard stop if equity drops 10% from peak
CONSECUTIVE_LOSS_LIMIT = 3      # Circuit breaker: pause after N losses in a row
ATR_STOP_MULTIPLIER  = 1.75     # Stop = 1.75x ATR from entry
ATR_PERIOD           = 14
ATR_TIMEFRAME        = "1H"

# ── Technical indicator parameters ───────────────────────────────────────────
EMA_FAST    = 9
EMA_MED     = 20
EMA_SLOW    = 50
EMA_TREND   = 200
RSI_PERIOD  = 21
RSI_OB      = 80                # Overbought (gold uses 80/20 not 70/30)
RSI_OS      = 20                # Oversold
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2.0

# ── SMC parameters ────────────────────────────────────────────────────────────
OB_LOOKBACK     = 20            # Candles to look back for order blocks
FVG_MIN_SIZE    = 0.5           # Minimum FVG size in USD
SWING_LOOKBACK  = 10            # Candles for swing high/low detection

# ── ML model ─────────────────────────────────────────────────────────────────
PREDICTION_HORIZON = 5          # Predict direction N candles ahead
TRAIN_SPLIT        = 0.75       # 75% train, 25% test
FEATURE_WINDOW     = 20         # Rolling window for feature calculation
MODEL_PATH         = "models/xgb_xauusd.joblib"

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2024-12-31"
INITIAL_CAPITAL = 10_000        # USD
COMMISSION      = 0.00007       # ~0.7 pips per side (spread simulation)

# ── MT5 broker ────────────────────────────────────────────────────────────────
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")
MT5_PATH     = os.getenv("MT5_PATH", "C:/Program Files/MetaTrader 5/terminal64.exe")

# ── Telegram alerts ───────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Macro data sources ────────────────────────────────────────────────────────
# DXY proxy on yfinance
DXY_TICKER      = "DX-Y.NYB"
# 10Y real yield (TIPS)
TIPS_10Y_TICKER = "^TYX"
# Fed Funds Rate proxy
FED_FUNDS_TICKER = "^IRX"
