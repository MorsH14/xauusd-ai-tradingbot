"""
MetaTrader 5 execution bridge.
Handles connection, order placement, position management, and account info.
Requires MetaTrader 5 terminal running on Windows with Python API installed.
"""
import time
from loguru import logger
from config.config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, SYMBOL_MT5
)


def _get_mt5():
    """Lazy import of MT5 — only available on Windows with terminal installed."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not found.\n"
            "Install with: pip install MetaTrader5\n"
            "Requires Windows + MT5 terminal installed."
        )


class MT5Bridge:
    def __init__(self):
        self.mt5 = _get_mt5()
        self.connected = False

    def connect(self) -> bool:
        """Initialize and login to MT5 terminal."""
        if not self.mt5.initialize(path=MT5_PATH):
            logger.error(f"MT5 initialize failed: {self.mt5.last_error()}")
            return False

        if not self.mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error(f"MT5 login failed: {self.mt5.last_error()}")
            self.mt5.shutdown()
            return False

        info = self.mt5.account_info()
        logger.info(
            f"MT5 connected | Account: {info.login} | Broker: {info.company} "
            f"| Balance: ${info.balance:.2f} | Equity: ${info.equity:.2f}"
        )
        self.connected = True
        return True

    def disconnect(self) -> None:
        if self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def get_account_info(self) -> dict:
        info = self.mt5.account_info()
        if info is None:
            return {}
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
            "profit":   info.profit,
            "leverage": info.leverage,
        }

    def get_symbol_info(self, symbol: str = SYMBOL_MT5) -> dict:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return {}
        return {
            "bid":        info.bid,
            "ask":        info.ask,
            "spread":     info.spread,
            "digits":     info.digits,
            "trade_mode": info.trade_mode,
        }

    def place_market_order(
        self,
        direction:   int,
        lot_size:    float,
        stop_loss:   float,
        take_profit: float,
        comment:     str = "XAUUSD_AI_BOT",
        symbol:      str = SYMBOL_MT5,
        magic:       int = 20240101,
    ) -> dict:
        """
        Place a market order.
        direction: 1 = BUY, -1 = SELL
        Returns order result dict.
        """
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Cannot get tick for {symbol}")
            return {"error": "No tick data"}

        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        price      = tick.ask if direction == 1 else tick.bid
        deviation  = 20  # Max slippage in points

        request = {
            "action":       self.mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot_size,
            "type":         order_type,
            "price":        price,
            "sl":           stop_loss,
            "tp":           take_profit,
            "deviation":    deviation,
            "magic":        magic,
            "comment":      comment,
            "type_time":    self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            error = self.mt5.last_error() if result is None else result.comment
            logger.error(f"Order failed: {error}")
            return {"error": error, "retcode": getattr(result, "retcode", -1)}

        logger.info(
            f"Order placed | {symbol} {'BUY' if direction == 1 else 'SELL'} "
            f"{lot_size} lots @ {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} "
            f"| Ticket: {result.order}"
        )
        return {"ticket": result.order, "price": result.price, "volume": result.volume}

    def close_all_positions(self, symbol: str = SYMBOL_MT5, magic: int = 20240101) -> int:
        """Close all open positions for this symbol/magic. Returns number closed."""
        positions = self.mt5.positions_get(symbol=symbol)
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic != magic:
                continue
            close_dir  = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
            tick       = self.mt5.symbol_info_tick(symbol)
            close_price = tick.bid if pos.type == 0 else tick.ask

            request = {
                "action":   self.mt5.TRADE_ACTION_DEAL,
                "symbol":   symbol,
                "volume":   pos.volume,
                "type":     close_dir,
                "position": pos.ticket,
                "price":    close_price,
                "deviation": 20,
                "magic":    magic,
                "comment":  "CLOSE_ALL",
            }
            result = self.mt5.order_send(request)
            if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                closed += 1
                logger.info(f"Closed position #{pos.ticket} | PnL: ${pos.profit:.2f}")

        return closed

    def get_open_positions(self, symbol: str = SYMBOL_MT5) -> list:
        positions = self.mt5.positions_get(symbol=symbol)
        if not positions:
            return []
        return [
            {
                "ticket":    p.ticket,
                "type":      "BUY" if p.type == 0 else "SELL",
                "volume":    p.volume,
                "open_price": p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "profit":    p.profit,
            }
            for p in positions
        ]
