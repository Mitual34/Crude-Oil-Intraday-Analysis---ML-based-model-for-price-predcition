"""Walk-forward intraday backtester driving the forecaster and portfolio."""
from __future__ import annotations

import traceback
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from .config import MAX_HOLD_PERIOD, STRATEGY_HORIZON
from .portfolio import PortfolioAnalyzer


class IntradayBacktester:
    def __init__(self, forecaster, train_days=15, test_days=5):
        self.forecaster = forecaster
        self.train_days = train_days
        self.test_days = test_days
        self.portfolio = PortfolioAnalyzer()
        self.trading_params = {
            'spread': 0.03, 'slippage_pct': 0.005,
            'threshold': 0.0005, 'risk_per_trade': 0.005
        }

    def backtest(self, intraday_data):
        end_date = intraday_data.index[-1]
        test_start = end_date - timedelta(days=self.test_days)
        intraday_data = intraday_data.sort_index()
        train_data = intraday_data[intraday_data.index < test_start]
        test_data = intraday_data[intraday_data.index >= test_start]

        min_train_bars = 100
        min_test_bars = 50
        if len(train_data) < min_train_bars or len(test_data) < min_test_bars:
            print(f"⚠️ Not enough data for backtest. Train: {len(train_data)} bars, Test: {len(test_data)} bars")
            return None, None, None, None

        print(f"  📊 Backtest data: Train bars: {len(train_data)}, Test bars: {len(test_data)}")

        features = self.forecaster.create_features(train_data.copy()).sort_index()
        if 'price' in features.columns:
            target = features['price']
            X_features = features.drop('price', axis=1)
        else:
            target = features.iloc[:, -1]
            X_features = features.iloc[:, :-1]

        self.forecaster.train_model(X_features, target)

        forecasts = []
        actuals = []
        dates = []
        directions = []
        current_day = None

        for i in range(len(test_data)):
            try:
                current_time = test_data.index[i]
                est = pytz.timezone('US/Eastern')
                utc_index = current_time.tz_localize('UTC') if current_time.tz is None else current_time
                est_index = utc_index.tz_convert(est)
                current_est_date = est_index.date()

                if current_day != current_est_date:
                    if current_day is not None:
                        self.portfolio.reset_daily(current_time)
                    current_day = current_est_date
                    print(f"  📅 Trading day: {current_day.strftime('%Y-%m-%d')}")

                current_data = intraday_data[intraday_data.index < current_time]
                min_bars_for_prediction = 50

                if len(current_data) < min_bars_for_prediction:
                    forecasts.append(np.nan)
                    actuals.append(float(test_data['price'].iloc[i]))
                    dates.append(current_time)
                    continue

                current_features = self.forecaster.create_features(current_data.copy()).sort_index()
                if 'price' in current_features.columns:
                    X_current = current_features.drop('price', axis=1)
                else:
                    X_current = current_features.iloc[:, :-1]

                try:
                    forecast = self.forecaster.forecast_prices(current_data)
                    if isinstance(forecast, pd.Series):
                        next_bar_forecast = float(forecast.iloc[0])
                    elif hasattr(forecast, 'item'):
                        next_bar_forecast = float(forecast.item())
                    else:
                        next_bar_forecast = float(forecast)
                except Exception as e:
                    print(f"  ⚠️ Forecast error: {e}")
                    next_bar_forecast = float(current_data['price'].iloc[-1])

                forecasts.append(next_bar_forecast)
                current_price = float(test_data['price'].iloc[i])
                actuals.append(current_price)
                dates.append(current_time)

                if i > 0 and not np.isnan(forecasts[i]) and not np.isnan(forecasts[i - 1]):
                    try:
                        prev_fc = float(forecasts[i - 1])
                        current_fc = float(forecasts[i])
                        prev_price = float(test_data['price'].iloc[i - 1])
                        current_price_val = float(current_price)
                        actual_direction = 1 if current_price_val > prev_price else 0
                        predicted_direction = 1 if current_fc > prev_fc else 0
                        directions.append(1 if actual_direction == predicted_direction else 0)
                    except Exception as e:
                        print(f"  ⚠️ Direction check error: {e}")

                signal = 0
                try:
                    current_price_val = float(current_price)
                    next_bar_forecast_val = float(next_bar_forecast)
                    threshold = self.trading_params['threshold']
                    upper_bound = current_price_val * (1 + threshold)
                    lower_bound = current_price_val * (1 - threshold)

                    if next_bar_forecast_val > upper_bound:
                        signal = 1
                    elif next_bar_forecast_val < lower_bound:
                        signal = -1
                except Exception as e:
                    print(f"  ⚠️ Signal generation error: {e}")

                close_due_to_time = False
                if self.portfolio.position != 0:
                    hold_time = current_time - self.portfolio.current_position['entry_time']
                    if hold_time > MAX_HOLD_PERIOD:
                        close_due_to_time = True
                        signal = 0

                if signal != self.portfolio.position or close_due_to_time:
                    if self.portfolio.position != 0:
                        self.portfolio.close_position(
                            current_price_val,
                            current_time,
                            spread=self.trading_params['spread'],
                            slippage_pct=self.trading_params['slippage_pct']
                        )

                    if signal != 0:
                        try:
                            position_size = float(self.forecaster.calculate_position_size(
                                current_data,
                                self.trading_params['risk_per_trade']
                            ))

                            entry_price = self.portfolio.execute_trade(
                                signal,
                                current_price_val,
                                spread=self.trading_params['spread'],
                                slippage_pct=self.trading_params['slippage_pct'],
                                position_size=position_size
                            )
                            self.portfolio.position = signal
                            self.portfolio.current_position = {
                                'entry_price': entry_price,
                                'size': position_size,
                                'signal': signal,
                                'entry_time': current_time
                            }
                            self.portfolio.trades.append({
                                'type': 'entry',
                                'price': entry_price,
                                'size': position_size,
                                'signal': signal,
                                'timestamp': current_time
                            })
                        except Exception as e:
                            print(f"Trade execution error: {e}")
            except Exception as e:
                print(f"⚠️ Error processing bar {i}: {str(e)}")
                traceback.print_exc()
                forecasts.append(np.nan)
                actuals.append(float(test_data['price'].iloc[i]))
                dates.append(current_time)
                signal = 0

        if self.portfolio.position != 0:
            self.portfolio.close_position(
                float(test_data['price'].iloc[-1]),
                test_data.index[-1],
                spread=self.trading_params['spread'],
                slippage_pct=self.trading_params['slippage_pct']
            )

        results = pd.DataFrame({'date': dates, 'actual': actuals, 'forecast': forecasts}).set_index('date')

        actual_values = np.array(actuals).ravel()
        forecast_values = np.array(forecasts).ravel()
        mask = ~np.isnan(forecast_values)

        if np.sum(mask) > 0:
            mae = mean_absolute_error(actual_values[mask], forecast_values[mask])
            rmse = np.sqrt(mean_squared_error(actual_values[mask], forecast_values[mask]))
            mape = mean_absolute_percentage_error(actual_values[mask], forecast_values[mask])
            accuracy = 100 * (1 - mape)
            if directions:
                directional_accuracy = sum(directions) / len(directions) * 100
            else:
                directional_accuracy = 0
        else:
            mae = rmse = mape = accuracy = directional_accuracy = 0

        accuracy_metrics = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'Accuracy': accuracy, 'Directional_Accuracy': directional_accuracy
        }

        trading_metrics = self.portfolio.calculate_metrics()
        return results, {**accuracy_metrics, **trading_metrics}, self.portfolio.trades, self.portfolio.equity_curve

    def plot_backtest(self, results, metrics):
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['actual'], label='Actual Prices', color='blue')
        plt.plot(results.index, results['forecast'], label='Forecast', color='red', linestyle='--')

        if hasattr(self.portfolio, 'trades') and self.portfolio.trades:
            for trade in self.portfolio.trades:
                if trade['type'] == 'entry':
                    marker = '^' if trade['signal'] == 1 else 'v'
                    color = 'green' if trade['signal'] == 1 else 'red'
                    plt.scatter(trade['timestamp'], trade['price'],
                                marker=marker, color=color, s=100, label='Entry' if trade['signal'] == 1 else None)
                elif trade['type'] == 'exit':
                    plt.scatter(trade['timestamp'], trade['price'],
                                marker='o', color='black', s=80, label='Exit')

        plt.title(f"Crude Oil {self.forecaster.interval} Backtest ({STRATEGY_HORIZON} horizon)")
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig('backtest_results.png')
        plt.close()
