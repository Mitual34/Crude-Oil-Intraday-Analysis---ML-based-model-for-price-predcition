"""Crude-oil price forecaster: data, features, model training and forecasting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from .config import EXPECTED_FEATURES, STRATEGY_HORIZON, VALID_INTERVALS
from .indicators import calculate_atr, calculate_macd, calculate_rsi
from .optional_deps import XGB_AVAILABLE, xgb


class CommodityForecaster:
    def __init__(self, interval='30m', forecast_bars=10):
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Valid options: {VALID_INTERVALS}")

        self.interval = interval
        self.forecast_bars = forecast_bars
        self.ticker = 'CL=F'
        self.usd_ticker = 'DX=F'

        if XGB_AVAILABLE:
            self.model = xgb.XGBRegressor(n_estimators=200, random_state=42)
            print("  ✅ Using XGBoost for forecasting")
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            print("  ⚠️ Using RandomForest as fallback")

        self.feature_columns = None
        self.required_min_bars = 30

    def get_historical_data(self, start_date, end_date):
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            oil_data = yf.download(
                self.ticker,
                start=start_str,
                end=end_str,
                interval=self.interval,
                prepost=True,
                progress=False,
                auto_adjust=True
            )

            if not oil_data.empty and (oil_data['Close'] <= 0).any().any():
                oil_data = oil_data[oil_data['Close'] > 0]

            if oil_data.empty:
                print("⚠️ No oil data returned - using fallback")
                return self.create_fallback_data(start_date, end_date)

            oil_data.index = oil_data.index.tz_localize(None)

            usd_data = yf.download(
                self.usd_ticker,
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True
            )

            if not usd_data.empty:
                usd_data.index = usd_data.index.tz_localize(None)
                usd_map = {}
                for date, row in usd_data.iterrows():
                    usd_map[date.date()] = row['Close']

                oil_data['usd_index'] = oil_data.index.map(
                    lambda x: usd_map.get(x.date(), np.nan)
                )
                oil_data['usd_index'] = oil_data['usd_index'].ffill()

            oil_data = oil_data.rename(columns={
                'Close': 'price', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'
            })

            oil_data = oil_data[['open', 'high', 'low', 'price', 'volume', 'usd_index']].ffill().dropna()
            oil_data = oil_data.sort_index()
            return oil_data

        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            return self.create_fallback_data(start_date, end_date)

    def create_fallback_data(self, start_date, end_date):
        print("⚠️ Generating fallback data")
        dates = pd.date_range(start=start_date, end=end_date, freq=self.interval)
        if len(dates) == 0:
            dates = pd.date_range(end=end_date, periods=100, freq=self.interval)

        base_price = 75.0
        prices = [base_price + i * 0.1 for i in range(len(dates))]

        df = pd.DataFrame({
            'price': prices,
            'open': [p - 0.1 for p in prices],
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.15 for p in prices],
            'volume': np.random.randint(10000, 50000, len(dates)),
            'usd_index': np.random.uniform(90, 110, len(dates))
        }, index=dates)
        return df.sort_index()

    def create_features(self, data, news_sentiment=None):
        required_columns = ['open', 'high', 'low', 'price', 'volume', 'usd_index']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 10000
                elif col == 'usd_index':
                    data[col] = 100.0
                else:
                    if 'price' not in data.columns and len(data.columns) > 0:
                        data = data.rename(columns={data.columns[0]: 'price'})
                    if 'price' in data.columns:
                        data[col] = data['price']
                    else:
                        data[col] = 0.0

        data = data.copy()
        data['rsi'] = calculate_rsi(data)
        macd_line, signal_line = calculate_macd(data)
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['atr'] = calculate_atr(data)

        data['volume_ma'] = data['volume'].rolling(20, min_periods=1).mean().fillna(data['volume'].mean() if not data.empty else 10000)
        data['volume_pct_change'] = data['volume'].pct_change().fillna(0)

        with np.errstate(divide='ignore', invalid='ignore'):
            returns = data['price'].pct_change().fillna(0)
            data['returns'] = returns

            price_ratio = data['price'] / data['price'].shift(1).replace(0, 1e-6)
            log_returns = np.log(price_ratio).fillna(0)
            data['log_returns'] = log_returns

        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['minute'] = data.index.minute
            data['day_of_week'] = data.index.dayofweek
            data['time_of_day'] = data['hour'] + data['minute'] / 60
        else:
            data['hour'] = 9
            data['minute'] = 30
            data['day_of_week'] = 0
            data['time_of_day'] = 9.5

        data['economic_event'] = 0

        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            est = pytz.timezone('US/Eastern')
            if data.index.tz is None:
                utc_index = data.index.tz_localize('UTC')
            else:
                utc_index = data.index
            est_index = utc_index.tz_convert(est)

            hour_arr = est_index.hour.values
            dayofweek_arr = est_index.dayofweek.values

            market_open_mask = pd.Series((hour_arr == 9) & (est_index.minute >= 30), index=data.index)
            inventory_report_mask = pd.Series((dayofweek_arr == 2) & (hour_arr == 10) & (est_index.minute >= 30), index=data.index)

            data.loc[market_open_mask, 'economic_event'] = 1
            data.loc[inventory_report_mask, 'economic_event'] = 2

        if news_sentiment is not None:
            data['news_sentiment'] = news_sentiment
        else:
            data['news_sentiment'] = 0.0

        for lag in [1, 2, 3, 5, 8]:
            col_name = f'returns_lag{lag}'
            data[col_name] = data['returns'].shift(lag).fillna(0)

        for feature in EXPECTED_FEATURES:
            if feature not in data.columns:
                data[feature] = 0.0

        # Handle infinite values safely: coerce to numeric, drop inf, fill gaps.
        if not data.empty:
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.ffill().fillna(0)
        return data

    def train_model(self, features, target):
        if features.empty or target.empty:
            raise ValueError("No data available for training")

        # Ensure data is float type
        features = features.astype(float)
        target = target.astype(float)

        self.feature_columns = features.columns.tolist()
        X = features.values
        y = target.values.ravel()

        # Check for and handle any remaining infinite values
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            print("⚠️ Data contains infinite values - replacing with NaN")
            X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
            y = np.nan_to_num(y, nan=0, posinf=1e10, neginf=-1e10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        try:
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = mean_absolute_percentage_error(y_test, predictions)

            if len(y_test) > 1:
                directional = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(predictions)))
                directional_percent = directional * 100
            else:
                directional_percent = 0

            model_type = "XGBoost" if XGB_AVAILABLE else "RandomForest"
            print(f"✅ {model_type} model trained - MAE: ${mae:.4f}, RMSE: ${rmse:.4f}, Directional: {directional_percent:.1f}%")
        except Exception as e:
            print(f"⚠️ Model training failed: {e} - falling back to RandomForest")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            model_type = "RandomForest (fallback)"
            predictions = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"✅ {model_type} model trained - MAE: ${mae:.4f}, RMSE: ${rmse:.4f}")

        return self.model

    def forecast_prices(self, data, news_sentiment=None):
        if data.empty:
            raise ValueError("No data available for forecasting")

        if self.feature_columns is None:
            raise RuntimeError("Model must be trained before forecasting")

        current_data = data.copy()
        last_timestamp = data.index[-1]

        interval_str = self.interval
        if interval_str.endswith('m') and not interval_str.endswith('min'):
            minutes = int(''.join(filter(str.isdigit, interval_str)))
        else:
            minutes = 30

        time_delta = pd.Timedelta(minutes=minutes)
        forecast_dates = [last_timestamp + (i + 1) * time_delta for i in range(self.forecast_bars)]
        forecasts = []

        for _ in range(self.forecast_bars):
            try:
                features = self.create_features(current_data, news_sentiment)
                if 'price' in features.columns:
                    X_current = features.drop('price', axis=1)
                else:
                    X_current = features.iloc[:, :-1]

                for col in self.feature_columns:
                    if col not in X_current.columns:
                        X_current[col] = 0.0

                X_current = X_current[self.feature_columns].iloc[[-1]]
                X_array = X_current.values.astype(float)
                next_price = self.model.predict(X_array)[0]
            except Exception as e:
                print(f"⚠️ Feature creation or prediction error: {e}")
                next_price = float(current_data['price'].iloc[-1])

            forecasts.append(next_price)
            new_bar = {
                'open': float(current_data['open'].iloc[-1]),
                'high': max(float(current_data['high'].iloc[-1]), next_price),
                'low': min(float(current_data['low'].iloc[-1]), next_price),
                'price': next_price,
                'volume': float(current_data['volume'].iloc[-1]) * 0.95,
                'usd_index': float(current_data['usd_index'].iloc[-1])
            }

            new_index = forecast_dates[len(forecasts) - 1]
            new_df = pd.DataFrame([new_bar], index=[new_index])
            current_data = pd.concat([current_data, new_df])

        if self.forecast_bars == 1:
            return float(forecasts[0])

        return pd.Series(forecasts, index=forecast_dates)

    def calculate_position_size(self, data, risk_per_trade=0.01):
        if len(data) < 10:
            return 1

        volatility = float(data['atr'].iloc[-1]) if 'atr' in data.columns else float(data['price'].std())
        if volatility == 0:
            return 1

        account_size = 10000
        position_size = (risk_per_trade * account_size) / volatility
        return max(0.5, min(position_size, 5))

    def plot_forecast(self, historical, forecast):
        plt.figure(figsize=(12, 6))
        plt.plot(historical.index, historical['price'], label='Historical Prices', color='blue')

        if isinstance(forecast, pd.Series):
            plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='--')
            start_price = float(forecast.iloc[0]) if not forecast.empty else 0
            end_price = float(forecast.iloc[-1]) if not forecast.empty else 0
        else:
            plt.scatter(historical.index[-1] + pd.Timedelta(minutes=30), forecast,
                        color='red', marker='o', s=100, label='Forecast')
            start_price = float(forecast)
            end_price = float(forecast)

        plt.title(f"Crude Oil {self.interval} Price Forecast ({STRATEGY_HORIZON} horizon)")
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig('oil_forecast.png')
        plt.close()
        return None, start_price, end_price
