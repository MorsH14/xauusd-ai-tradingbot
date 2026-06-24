"""
Telegram alert system and performance tracker.
Sends real-time notifications for trade entries, exits, drawdowns, and daily summaries.
"""
import requests
from datetime import datetime
from loguru import logger
from config.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def _send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping alert")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")
        return False


def alert_trade_open(
    direction:   int,
    symbol:      str,
    entry:       float,
    stop_loss:   float,
    take_profit: float,
    lot_size:    float,
    confidence:  float,
    regime:      str,
) -> None:
    emoji  = "🟢 BUY" if direction == 1 else "🔴 SELL"
    rr     = abs(take_profit - entry) / abs(stop_loss - entry) if abs(stop_loss - entry) > 0 else 0
    msg = (
        f"<b>{emoji} TRADE OPENED</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"Symbol:      <code>{symbol}</code>\n"
        f"Entry:       <code>{entry:.2f}</code>\n"
        f"Stop Loss:   <code>{stop_loss:.2f}</code>\n"
        f"Take Profit: <code>{take_profit:.2f}</code>\n"
        f"Lot Size:    <code>{lot_size}</code>\n"
        f"R:R Ratio:   <code>1:{rr:.1f}</code>\n"
        f"Confidence:  <code>{confidence:.1%}</code>\n"
        f"Regime:      <code>{regime}</code>\n"
        f"Time (UTC):  <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}</code>"
    )
    _send_telegram(msg)
    logger.info(f"Alert sent: Trade Open {emoji} @ {entry:.2f}")


def alert_trade_close(
    direction:   int,
    symbol:      str,
    entry:       float,
    exit_price:  float,
    pnl_usd:     float,
    exit_reason: str,
    equity:      float,
) -> None:
    emoji = "✅" if pnl_usd > 0 else "❌"
    side  = "BUY" if direction == 1 else "SELL"
    msg = (
        f"<b>{emoji} TRADE CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"Symbol:     <code>{symbol}</code>\n"
        f"Direction:  <code>{side}</code>\n"
        f"Entry:      <code>{entry:.2f}</code>\n"
        f"Exit:       <code>{exit_price:.2f}</code>\n"
        f"PnL:        <b><code>${pnl_usd:+.2f}</code></b>\n"
        f"Reason:     <code>{exit_reason}</code>\n"
        f"Equity:     <code>${equity:.2f}</code>\n"
        f"Time (UTC): <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}</code>"
    )
    _send_telegram(msg)
    logger.info(f"Alert sent: Trade Close {emoji} PnL=${pnl_usd:+.2f}")


def alert_circuit_breaker(reason: str, equity: float) -> None:
    msg = (
        f"<b>⚠️ CIRCUIT BREAKER TRIGGERED</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"Reason:  <code>{reason}</code>\n"
        f"Equity:  <code>${equity:.2f}</code>\n"
        f"Time:    <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</code>\n\n"
        f"<i>Bot has paused trading.</i>"
    )
    _send_telegram(msg)
    logger.warning(f"Circuit breaker alert sent: {reason}")


def alert_daily_summary(metrics: dict) -> None:
    pnl    = metrics.get("daily_pnl_usd", 0)
    emoji  = "📈" if pnl >= 0 else "📉"
    msg = (
        f"<b>{emoji} DAILY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"Date:         <code>{datetime.utcnow().strftime('%Y-%m-%d')}</code>\n"
        f"Daily PnL:    <b><code>${pnl:+.2f}</code></b>\n"
        f"Equity:       <code>${metrics.get('equity', 0):.2f}</code>\n"
        f"Trades Today: <code>{metrics.get('trades_today', 0)}</code>\n"
        f"Win Rate:     <code>{metrics.get('win_rate_pct', 0):.1f}%</code>\n"
        f"Drawdown:     <code>{metrics.get('total_drawdown_pct', 0):.1f}%</code>"
    )
    _send_telegram(msg)
    logger.info("Daily summary alert sent")


def alert_error(error_msg: str) -> None:
    msg = (
        f"<b>🚨 BOT ERROR</b>\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"<code>{error_msg[:500]}</code>\n"
        f"Time: <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</code>"
    )
    _send_telegram(msg)
    logger.error(f"Error alert sent: {error_msg}")
