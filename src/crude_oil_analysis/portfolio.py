"""Portfolio accounting and performance metrics for the intraday backtest."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import STRATEGY_HORIZON


class PortfolioAnalyzer:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades = []
        self.drawdowns = []
        self.returns = []
        self.current_day_equity = initial_capital
        self.position = 0
        self.current_position = None

    def execute_trade(self, signal, entry_price, spread=0.05, slippage_pct=0.01, position_size=1):
        slippage = entry_price * (slippage_pct / 100)
        entry_price_adjusted = entry_price + (spread / 2) + slippage
        return entry_price_adjusted

    def close_position(self, exit_price, exit_timestamp, spread=0.05, slippage_pct=0.01):
        if not self.current_position:
            return 0

        slippage = exit_price * (slippage_pct / 100)
        exit_price_adjusted = exit_price - (spread / 2) - slippage

        if self.position == 1:
            profit = (exit_price_adjusted - self.current_position['entry_price']) * self.current_position['size']
        elif self.position == -1:
            profit = (self.current_position['entry_price'] - exit_price_adjusted) * self.current_position['size']
        else:
            profit = 0

        hold_time = (exit_timestamp - self.current_position['entry_time']).total_seconds() / 60

        self.update_portfolio(profit)
        self.trades.append({
            'type': 'exit',
            'price': exit_price,
            'profit': profit,
            'equity': self.equity_curve[-1],
            'timestamp': exit_timestamp,
            'hold_time': hold_time
        })

        self.position = 0
        self.current_position = None
        return profit

    def update_portfolio(self, profit):
        self.current_day_equity += profit
        current_equity = self.equity_curve[-1] + profit
        self.equity_curve.append(current_equity)
        self.returns.append(profit / self.equity_curve[-2] if self.equity_curve[-2] != 0 else 0)

        peak = max(self.equity_curve)
        current_dd = (peak - current_equity) / peak if peak > 0 else 0
        self.drawdowns.append(current_dd)

    def reset_daily(self, reset_timestamp):
        if self.position != 0:
            self.close_position(
                self.current_position['entry_price'] * (1.01 if self.position == 1 else 0.99),
                reset_timestamp
            )
        self.current_day_equity = self.equity_curve[-1]

    def calculate_metrics(self, risk_free_rate=0.02):
        if not self.returns:
            return {
                'total_return': 0, 'max_drawdown': 0, 'win_loss_ratio': 0,
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'avg_trade_duration': 0,
                'profit_factor': 0, 'win_rate': 0
            }

        returns = pd.Series(self.returns)
        total_return = (self.equity_curve[-1] / self.initial_capital - 1) * 100
        max_drawdown = max(self.drawdowns) * 100 if self.drawdowns else 0

        exit_trades = [t for t in self.trades if t['type'] == 'exit']
        winning_trades = [t for t in exit_trades if t['profit'] > 0]
        losing_trades = [t for t in exit_trades if t['profit'] < 0]
        win_loss_ratio = len(winning_trades) / len(losing_trades) if losing_trades else float('inf')

        gross_profit = sum(t['profit'] for t in winning_trades)
        gross_loss = abs(sum(t['profit'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        if exit_trades:
            durations = [t['hold_time'] for t in exit_trades]
            avg_trade_duration = sum(durations) / len(durations)
        else:
            avg_trade_duration = 0

        annual_factor = np.sqrt(252 * 78)
        excess_returns = returns - risk_free_rate / (252 * 78)

        if excess_returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * annual_factor

        downside_returns = returns[returns < 0]
        if downside_returns.empty:
            sortino_ratio = 0
        else:
            downside_volatility = downside_returns.std()
            sortino_ratio = (returns.mean() - risk_free_rate / (252 * 78)) / downside_volatility * annual_factor

        win_rate = len(winning_trades) / len(exit_trades) if exit_trades else 0

        return {
            'total_return': total_return, 'max_drawdown': max_drawdown,
            'win_loss_ratio': win_loss_ratio, 'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio,
            'avg_trade_duration': avg_trade_duration, 'win_rate': win_rate
        }

    def plot_equity_curve(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Value')
        plt.title(f"Intraday Portfolio Equity Curve ({STRATEGY_HORIZON} horizon)")
        plt.ylabel("Value ($)")
        plt.xlabel("Trade")
        plt.legend()
        plt.grid(True)
        plt.savefig('intraday_portfolio_equity.png')
        plt.close()
