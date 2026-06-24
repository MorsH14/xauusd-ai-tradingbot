"""
Risk management engine.
Handles position sizing, drawdown tracking, and exposure limits.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from config.config import (
    RISK_PER_TRADE, MAX_DAILY_DRAWDOWN, MAX_TOTAL_DRAWDOWN,
    CONSECUTIVE_LOSS_LIMIT, ATR_STOP_MULTIPLIER,
)


@dataclass
class AccountState:
    equity:          float
    peak_equity:     float = field(init=False)
    daily_start:     float = field(init=False)
    consecutive_losses: int = 0
    total_trades:    int = 0
    winning_trades:  int = 0
    open_position:   bool = False

    def __post_init__(self):
        self.peak_equity = self.equity
        self.daily_start = self.equity

    @property
    def total_drawdown(self) -> float:
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_drawdown(self) -> float:
        return (self.daily_start - self.equity) / self.daily_start

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

    def update_after_trade(self, pnl: float) -> None:
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def reset_daily(self) -> None:
        self.daily_start = self.equity


class RiskManager:
    def __init__(self, initial_equity: float):
        self.account = AccountState(equity=initial_equity)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss:   float,
        pip_value:   float = 0.01,   # USD per pip per 0.01 lot (XAUUSD standard)
    ) -> float:
        """
        ATR-based position sizing.
        Risk exactly RISK_PER_TRADE * equity on each trade.
        Returns lot size (rounded to 2 decimal places).
        """
        risk_amount    = self.account.equity * RISK_PER_TRADE
        stop_distance  = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            logger.warning("Stop distance is 0 — cannot size position")
            return 0.0

        # For XAUUSD: 1 lot = 100 oz, pip = $0.01, so $1 per pip per 0.01 lot
        # stop_distance in price = stop_distance / pip_value pips
        pips_at_risk   = stop_distance / pip_value
        lot_size       = risk_amount / (pips_at_risk * pip_value * 100)
        lot_size       = round(max(0.01, min(lot_size, 10.0)), 2)  # Clamp 0.01–10 lots

        logger.debug(f"Position size: {lot_size} lots | Risk: ${risk_amount:.2f} | Stop: {stop_distance:.2f}")
        return lot_size

    def can_trade(self) -> tuple[bool, str]:
        """Check all circuit breakers before allowing a new trade."""
        if self.account.open_position:
            return False, "Already in a position"

        if self.account.total_drawdown >= MAX_TOTAL_DRAWDOWN:
            return False, f"Total drawdown {self.account.total_drawdown:.1%} exceeds limit {MAX_TOTAL_DRAWDOWN:.1%}"

        if self.account.daily_drawdown >= MAX_DAILY_DRAWDOWN:
            return False, f"Daily drawdown {self.account.daily_drawdown:.1%} exceeds limit {MAX_DAILY_DRAWDOWN:.1%}"

        if self.account.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            return False, f"Circuit breaker: {self.account.consecutive_losses} consecutive losses"

        return True, "OK"

    def record_trade(self, pnl: float) -> None:
        self.account.update_after_trade(pnl)
        logger.info(
            f"Trade closed | PnL: ${pnl:+.2f} | Equity: ${self.account.equity:.2f} "
            f"| DrawdownTotal: {self.account.total_drawdown:.1%} "
            f"| WinRate: {self.account.win_rate:.1%}"
        )

    def reset_daily(self) -> None:
        self.account.reset_daily()
        logger.info(f"Daily reset | Starting equity: ${self.account.daily_start:.2f}")

    def summary(self) -> dict:
        a = self.account
        return {
            "equity":             round(a.equity, 2),
            "peak_equity":        round(a.peak_equity, 2),
            "total_drawdown_pct": round(a.total_drawdown * 100, 2),
            "daily_drawdown_pct": round(a.daily_drawdown * 100, 2),
            "total_trades":       a.total_trades,
            "win_rate_pct":       round(a.win_rate * 100, 2),
            "consecutive_losses": a.consecutive_losses,
        }
