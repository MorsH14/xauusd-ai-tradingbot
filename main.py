"""
XAUUSD AI Trading Bot — Live trading entry point.
Run: python main.py

Requires:
  - MetaTrader 5 terminal running
  - .env file with MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
  - Trained model at models/xgb_xauusd.joblib (run models/trainer.py first)
"""
import os
import time
import schedule
from datetime import datetime, timezone
from loguru import logger

from config.config import SYMBOL_MT5, MODEL_PATH
from data.fetcher import fetch_mt5_ohlcv
from data.preprocessor import prepare_dataset
from features.pipeline import build_features, get_feature_matrix
from models.classifier import GoldDirectionClassifier
from strategy.signal import generate_signal
from strategy.regime import detect_regime
from risk.manager import RiskManager
from execution.mt5_bridge import MT5Bridge
from monitoring.alerts import alert_trade_open, alert_trade_close, alert_circuit_breaker, alert_daily_summary, alert_error
from config.config import INITIAL_CAPITAL

# ── Setup logging ──────────────────────────────────────────────────────────────
logger.add("logs/bot_{time}.log", rotation="1 day", retention="30 days", level="INFO")

# ── Global state ───────────────────────────────────────────────────────────────
clf          = GoldDirectionClassifier()
risk_manager = RiskManager(INITIAL_CAPITAL)
bridge       = MT5Bridge()
current_trade = None


def initialize():
    """Load model and connect to MT5."""
    logger.info("=== XAUUSD AI Bot Initializing ===")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run 'python models/trainer.py' first.")
        raise FileNotFoundError(f"No model at {MODEL_PATH}")

    clf.load(MODEL_PATH)
    logger.info("Model loaded")

    if not bridge.connect():
        raise ConnectionError("Failed to connect to MetaTrader 5")

    info = bridge.get_account_info()
    logger.info(f"Account balance: ${info.get('balance', 0):.2f}")


def on_new_bar():
    """Main trading loop — called on every new H1 bar close."""
    global current_trade

    now = datetime.now(timezone.utc)
    logger.info(f"New bar: {now.strftime('%Y-%m-%d %H:%M')} UTC")

    try:
        # 1. Fetch latest bars
        import MetaTrader5 as mt5
        df_raw = fetch_mt5_ohlcv(SYMBOL_MT5, mt5.TIMEFRAME_H1, n_bars=500)
        df     = prepare_dataset(df_raw)
        df     = build_features(df, include_target=False)

        if df.empty:
            logger.warning("No data after preprocessing")
            return

        # 2. Get ML probability for latest bar
        X, _ = get_feature_matrix(df)
        if X.empty:
            return

        latest_X     = X.iloc[[-1]]
        latest_row   = df.iloc[-1]
        ml_proba     = float(clf.predict_proba(latest_X)[0])
        regimes      = detect_regime(df)
        latest_regime = regimes.iloc[-1]

        # 3. Check if we can trade
        can_trade, reason = risk_manager.can_trade()
        if not can_trade:
            logger.info(f"No trade: {reason}")
            return

        # 4. Generate signal
        signal = generate_signal(latest_row, ml_proba, latest_regime)

        if signal.direction == 0:
            logger.info(f"No signal: {signal.reason[:80]}")
            return

        # 5. Calculate position size
        lot_size = risk_manager.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
        )

        if lot_size <= 0:
            logger.warning("Position size calculation returned 0 — skipping")
            return

        # 6. Place order
        result = bridge.place_market_order(
            direction=signal.direction,
            lot_size=lot_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if "error" not in result:
            current_trade = {
                "ticket":     result["ticket"],
                "direction":  signal.direction,
                "entry":      signal.entry_price,
                "stop_loss":  signal.stop_loss,
                "take_profit": signal.take_profit,
                "lot_size":   lot_size,
            }
            alert_trade_open(
                direction=signal.direction,
                symbol=SYMBOL_MT5,
                entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                lot_size=lot_size,
                confidence=signal.confidence,
                regime=signal.regime,
            )
            risk_manager.account.open_position = True

    except Exception as e:
        logger.exception(f"Error in on_new_bar: {e}")
        alert_error(str(e))


def send_daily_summary():
    """Send daily performance summary via Telegram."""
    summary = risk_manager.summary()
    summary["daily_pnl_usd"] = summary["equity"] - INITIAL_CAPITAL
    summary["trades_today"]  = summary["total_trades"]
    alert_daily_summary(summary)
    risk_manager.reset_daily()


def main():
    initialize()

    # Schedule the bot to check every hour at :01 (just after bar close)
    schedule.every().hour.at(":01").do(on_new_bar)
    schedule.every().day.at("21:05").do(send_daily_summary)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    on_new_bar()  # Run immediately on start

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        bridge.disconnect()


if __name__ == "__main__":
    main()
