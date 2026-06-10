"""Tests for the portfolio accounting and metrics."""
from datetime import datetime, timedelta

from crude_oil_analysis.portfolio import PortfolioAnalyzer

METRIC_KEYS = {
    "total_return", "max_drawdown", "win_loss_ratio", "profit_factor",
    "sharpe_ratio", "sortino_ratio", "avg_trade_duration", "win_rate",
}


def test_metrics_keys_on_empty_portfolio():
    assert set(PortfolioAnalyzer().calculate_metrics().keys()) == METRIC_KEYS


def test_winning_long_trade_increases_equity():
    p = PortfolioAnalyzer(initial_capital=10000)
    t0 = datetime(2024, 1, 1, 9, 30)
    entry = p.execute_trade(signal=1, entry_price=75.0, position_size=1)
    p.position = 1
    p.current_position = {"entry_price": entry, "size": 1, "signal": 1, "entry_time": t0}

    profit = p.close_position(80.0, t0 + timedelta(minutes=5))
    assert profit > 0
    assert p.equity_curve[-1] > 10000
    assert p.position == 0 and p.current_position is None


def test_metrics_after_a_trade_are_well_formed():
    p = PortfolioAnalyzer()
    t0 = datetime(2024, 1, 1, 9, 30)
    entry = p.execute_trade(signal=1, entry_price=75.0)
    p.position = 1
    p.current_position = {"entry_price": entry, "size": 1, "signal": 1, "entry_time": t0}
    p.close_position(76.0, t0 + timedelta(minutes=3))

    m = p.calculate_metrics()
    assert set(m.keys()) == METRIC_KEYS
    assert m["win_rate"] == 1.0
