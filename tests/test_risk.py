"""Tests for the risk management engine."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.manager import RiskManager, AccountState


class TestAccountState:
    def test_initial_peak_equals_equity(self):
        acc = AccountState(equity=10000)
        assert acc.peak_equity == 10000

    def test_drawdown_zero_at_start(self):
        acc = AccountState(equity=10000)
        assert acc.total_drawdown == 0.0

    def test_drawdown_after_loss(self):
        acc = AccountState(equity=10000)
        acc.update_after_trade(-1000)
        assert abs(acc.total_drawdown - 0.10) < 1e-6

    def test_consecutive_loss_counter(self):
        acc = AccountState(equity=10000)
        acc.update_after_trade(-100)
        acc.update_after_trade(-100)
        assert acc.consecutive_losses == 2

    def test_consecutive_loss_resets_on_win(self):
        acc = AccountState(equity=10000)
        acc.update_after_trade(-100)
        acc.update_after_trade(-100)
        acc.update_after_trade(300)
        assert acc.consecutive_losses == 0

    def test_win_rate(self):
        acc = AccountState(equity=10000)
        acc.update_after_trade(100)
        acc.update_after_trade(-50)
        acc.update_after_trade(200)
        assert abs(acc.win_rate - 2/3) < 1e-6


class TestRiskManager:
    def test_can_trade_initially(self):
        rm = RiskManager(10000)
        ok, _ = rm.can_trade()
        assert ok

    def test_circuit_breaker_consecutive_losses(self):
        rm = RiskManager(10000)
        for _ in range(3):
            rm.record_trade(-100)
        ok, reason = rm.can_trade()
        assert not ok
        assert "consecutive" in reason.lower()

    def test_circuit_breaker_total_drawdown(self):
        rm = RiskManager(10000)
        rm.record_trade(-1500)  # 15% loss — exceeds 10% limit
        ok, reason = rm.can_trade()
        assert not ok
        assert "drawdown" in reason.lower()

    def test_position_size_scales_with_equity(self):
        rm = RiskManager(10000)
        size1 = rm.calculate_position_size(1800, 1790)   # 10 USD stop

        rm2 = RiskManager(20000)
        size2 = rm2.calculate_position_size(1800, 1790)

        assert size2 > size1  # Larger equity = larger position

    def test_position_size_minimum(self):
        rm = RiskManager(100)   # Very small account
        size = rm.calculate_position_size(2000, 1900)   # Wide stop
        assert size >= 0.01

    def test_daily_reset(self):
        rm = RiskManager(10000)
        rm.record_trade(-200)
        rm.reset_daily()
        assert rm.account.daily_start == rm.account.equity
