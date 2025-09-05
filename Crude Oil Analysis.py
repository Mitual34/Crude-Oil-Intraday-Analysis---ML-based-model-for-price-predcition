import feedparser
from bs4 import BeautifulSoup
import urllib.parse
from dateparser import parse as parse_date
import requests
from datetime import datetime, timedelta
import numpy as np
import importlib.metadata
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pytz
import argparse
import sys
import warnings
import traceback
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import tempfile

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Strategy Horizon Configuration
STRATEGY_HORIZON = "intraday"
VALID_INTERVALS = ['5m', '15m', '30m', '60m']
MAX_HOLD_PERIOD = timedelta(minutes=5)

# Define expected features
EXPECTED_FEATURES = [
    'rsi', 'macd', 'macd_signal', 'atr', 'volume_ma', 'volume_pct_change',
    'returns', 'log_returns', 'hour', 'minute', 'day_of_week', 'time_of_day',
    'economic_event', 'news_sentiment',
    'returns_lag1', 'returns_lag2', 'returns_lag3', 'returns_lag5', 'returns_lag8'
]

# Enhanced XGBoost installation handling
print("🔍 Checking for XGBoost installation...")
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost is available - using for modeling")
except ImportError:
    print("⚠️ XGBoost not found - attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        import xgboost as xgb
        XGB_AVAILABLE = True
        print("✅ XGBoost installed successfully - using for modeling")
    except:
        print("❌ XGBoost installation failed - falling back to RandomForest")
        XGB_AVAILABLE = False

# Enhanced feedparser installation handling
print("🔍 Checking for feedparser installation...")
FEEDPARSER_AVAILABLE = False
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
    print("✅ feedparser is available")
except ImportError:
    print("⚠️ feedparser not found - attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "feedparser"])
        import feedparser
        FEEDPARSER_AVAILABLE = True
        print("✅ feedparser installed successfully")
    except:
        print("❌ feedparser installation failed - news features disabled")
        FEEDPARSER_AVAILABLE = False

# If feedparser still not available, create a dummy class
if not FEEDPARSER_AVAILABLE:
    print("⚠️ Using dummy feedparser class - news features disabled")
    class DummyFeedparser:
        def parse(self, *args, **kwargs):
            return {'entries': []}
    feedparser = DummyFeedparser()

# Enhanced FPDF2 installation handling
print("🔍 Checking for FPDF2 installation...")
PDF_AVAILABLE = False
FPDF_CLASS = None
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    PDF_AVAILABLE = True
    FPDF_CLASS = FPDF
    print("✅ FPDF2 is available - PDF reports enabled")
except ImportError:
    print("⚠️ FPDF2 not found - attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
        PDF_AVAILABLE = True
        FPDF_CLASS = FPDF
        print("✅ FPDF2 installed successfully - PDF reports enabled")
    except:
        print("❌ FPDF2 installation failed - PDF reports disabled")
        PDF_AVAILABLE = False

# If FPDF still not available, create a dummy class
if not PDF_AVAILABLE:
    print("⚠️ Using dummy FPDF class - PDF reports disabled")
    class DummyFPDF:
        def add_page(self): pass
        def set_font(self, *args, **kwargs): pass
        def cell(self, w=0, h=0, text='', border=0, new_x=XPos.RIGHT, new_y=YPos.NEXT, align='', fill=False, link=''): pass
        def ln(self, *args, **kwargs): pass
        def multi_cell(self, w=0, h=0, text='', border=0, align='J', fill=False, max_line_height=0): pass
        def image(self, *args, **kwargs): pass
        def get_y(self): return 0
        def output(self, *args, **kwargs): pass
        def set_margins(self, *args, **kwargs): pass
    FPDF_CLASS = DummyFPDF

# Get TextBlob version safely
try:
    TEXTBLOB_VERSION = importlib.metadata.version('textblob')
except Exception:
    TEXTBLOB_VERSION = "unknown"

class GoogleNews:
    def __init__(self, lang='en', country='US'):
        self.lang = lang.lower()
        self.country = country.upper()
        self.BASE_URL = 'https://news.google.com/rss'
    
    def search(self, query, from_=None, to_=None, exclude=None):
        # Only attempt news collection if feedparser is available
        if not FEEDPARSER_AVAILABLE:
            return {'entries': []}

        params = {
            'q': query,
            'hl': f'{self.lang}-{self.country}',
            'gl': self.country,
            'ceid': f'{self.country}:{self.lang}'
        }
        
        if from_ or to_:
            date_range = []
            if from_: date_range.append(f'after:{from_.replace("-", "/")}')
            if to_: date_range.append(f'before:{to_.replace("-", "/")}')
            params['q'] += ' ' + ' '.join(date_range)
        
        if exclude:
            if isinstance(exclude, list):
                exclude_str = ' '.join([f'-{term}' for term in exclude])
            else:
                exclude_str = f'-{exclude}'
            params['q'] += ' ' + exclude_str
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        feed = feedparser.parse(url)
        return feed

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
        entry_price_adjusted = entry_price + (spread/2) + slippage
        return entry_price_adjusted
        
    def close_position(self, exit_price, exit_timestamp, spread=0.05, slippage_pct=0.01):
        if not self.current_position:
            return 0
            
        slippage = exit_price * (slippage_pct / 100)
        exit_price_adjusted = exit_price - (spread/2) - slippage
        
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
        prices = [base_price + i*0.1 for i in range(len(dates))]
        
        df = pd.DataFrame({
            'price': prices,
            'open': [p - 0.1 for p in prices],
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.15 for p in prices],
            'volume': np.random.randint(10000, 50000, len(dates)),
            'usd_index': np.random.uniform(90, 110, len(dates))
        }, index=dates)
        return df.sort_index()
    
    def calculate_rsi(self, data, window=14):
        if len(data) < 2:
            return pd.Series([50] * len(data), index=data.index)
            
        delta = data['price'].diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean().fillna(0)
        avg_loss = loss.rolling(window=window, min_periods=1).mean().fillna(0)
        
        avg_loss = avg_loss.replace(0, 0.001)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        if len(data) < slow:
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
            
        exp1 = data['price'].ewm(span=fast, adjust=False).mean()
        exp2 = data['price'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0), signal_line.fillna(0)
    
    def calculate_atr(self, data, window=14):
        if len(data) < 2:
            return pd.Series(0, index=data.index)
            
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['price'].shift().fillna(method='bfill'))
        low_close = np.abs(data['low'] - data['price'].shift().fillna(method='bfill'))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr.fillna(0)
    
    def create_features(self, data, news_sentiment=None):
        required_columns = ['open', 'high', 'low', 'price', 'volume', 'usd_index']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume': data[col] = 10000
                elif col == 'usd_index': data[col] = 100.0
                else:
                    if 'price' not in data.columns and len(data.columns) > 0:
                        data = data.rename(columns={data.columns[0]: 'price'})
                    if 'price' in data.columns:
                        data[col] = data['price']
                    else:
                        data[col] = 0.0
        
        data = data.copy()
        data['rsi'] = self.calculate_rsi(data)
        macd_line, signal_line = self.calculate_macd(data)
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['atr'] = self.calculate_atr(data)
        
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
            data['time_of_day'] = data['hour'] + data['minute']/60
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
        
        # FIXED: Handle infinite values safely
        # Convert to float if needed and replace inf with nan
        if not data.empty:
            # Convert all columns to numeric, coercing errors to NaN
            data = data.apply(pd.to_numeric, errors='coerce')
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            # Forward fill and then fill remaining with 0
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
        
        # Fix: Check for and handle any remaining infinite values
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            print("⚠️ Data contains infinite values - replacing with NaN")
            X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
            y = np.nan_to_num(y, nan=0, posinf=1e10, neginf=-1e10)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Fix: Handle XGBoost training failure
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
        forecast_dates = [last_timestamp + (i+1)*time_delta for i in range(self.forecast_bars)]
        forecasts = []
        
        for _ in range(self.forecast_bars):
            try:
                # FIXED: Changed from self.forecaster to self
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
            
            new_index = forecast_dates[len(forecasts)-1]
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


def analyze_sentiment(headlines):
    if not headlines:
        return 0.0
        
    sentiments = []
    for title in headlines:
        try:
            analysis = TextBlob(title)
            sentiments.append(analysis.sentiment.polarity)
        except Exception:
            continue
    
    if sentiments:
        return round(sum(sentiments) / len(sentiments), 2)
    return 0.0


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
                
                if i > 0 and not np.isnan(forecasts[i]) and not np.isnan(forecasts[i-1]):
                    try:
                        prev_fc = float(forecasts[i-1])
                        current_fc = float(forecasts[i])
                        prev_price = float(test_data['price'].iloc[i-1])
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
                    
                    if next_bar_forecast_val > upper_bound: signal = 1
                    elif next_bar_forecast_val < lower_bound: signal = -1
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


def analyze_crude_oil_intraday(start_date, end_date, interval='30m', forecast_bars=12, backtest=True):
    print(f"\n🔍 Starting intraday analysis for Crude Oil ({interval} bars) [{STRATEGY_HORIZON} horizon]")
    print(f"  📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    news_client = GoogleNews()
    to_date = datetime.now()
    from_date = to_date - timedelta(days=1)
    
    headlines = []
    sentiment = 0.0
    try:
        articles = news_client.search(
            query='crude oil OR WTI OR OPEC',
            from_=from_date.strftime('%Y-%m-%d'),
            to_=to_date.strftime('%Y-%m-%d')
        )
        headlines = [entry['title'] for entry in articles.get('entries', [])[:20]]
        sentiment = analyze_sentiment(headlines)
        print(f"  📰 News Analysis: {len(headlines)} articles | Sentiment: {sentiment:.2f}")
    except Exception as e:
        print(f"  ❌ News collection failed: {str(e)}")
    
    forecaster = CommodityForecaster(interval=interval, forecast_bars=forecast_bars)
    
    try:
        intraday_data = forecaster.get_historical_data(start_date, end_date)
        if intraday_data is None or intraday_data.empty:
            print("  ❌ Historical data retrieval failed - using fallback")
            intraday_data = forecaster.create_fallback_data(start_date, end_date)
            
        if not intraday_data.empty:
            latest_price_val = float(intraday_data['price'].iloc[-1])
        else:
            latest_price_val = 0.0

        print(f"  📈 Historical Data: {len(intraday_data)} {interval} bars | Latest price: ${latest_price_val:.2f}")
        
        backtest_results = None
        backtest_accuracy = None
        trading_metrics = None
        equity_curve = None
        
        min_bars_for_backtest = 150
        if backtest and len(intraday_data) > min_bars_for_backtest:
            print(f"  🔄 Running intraday backtest ({interval} bars) [{STRATEGY_HORIZON} horizon]...")
            backtester = IntradayBacktester(forecaster, train_days=15, test_days=5)
            backtest_results, metrics, trades, equity_curve = backtester.backtest(intraday_data)
            
            if backtest_results is not None:
                backtest_accuracy = metrics.get('Accuracy', 0)
                trading_metrics = {
                    'total_return': metrics.get('total_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_loss_ratio': metrics.get('win_loss_ratio', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'avg_trade_duration': metrics.get('avg_trade_duration', 0)
                }
                print(f"  ✅ Backtest complete | Trades: {len(trades)//2 if trades else 0}")
                print(f"  📊 Trading Performance ({STRATEGY_HORIZON} horizon):")
                print(f"     - Total Return: {trading_metrics['total_return']:.2f}%")
                print(f"     - Win Rate: {trading_metrics['win_rate']*100:.1f}%")
                print(f"     - Avg Trade Duration: {trading_metrics['avg_trade_duration']:.1f} min")
                print(f"  📈 Model Metrics:")
                print(f"     - MAE: ${metrics.get('MAE', 0):.4f}")
                print(f"     - RMSE: ${metrics.get('RMSE', 0):.4f}")
                print(f"     - Directional Accuracy: {metrics.get('Directional_Accuracy', 0):.1f}%")
        else:
            print(f"  ⚠️ Skipping backtest - insufficient historical data ({len(intraday_data)} < {min_bars_for_backtest})")
        
        features = forecaster.create_features(intraday_data.copy(), sentiment)
        if 'price' in features.columns:
            X_features = features.drop('price', axis=1)
            y_target = features['price']
        else:
            X_features = features.iloc[:, :-1]
            y_target = features.iloc[:, -1]
        
        forecaster.train_model(X_features, y_target)
        forecast_series = forecaster.forecast_prices(intraday_data, sentiment)
        _, start_price, end_price = forecaster.plot_forecast(intraday_data, forecast_series)
        
        price_change = end_price - start_price
        change_pct = (end_price / start_price - 1) * 100 if start_price != 0 else 0
        
        print(f"  🔮 {forecast_bars}-Bar Forecast: "
              f"Start: ${start_price:.2f} | "
              f"End: ${end_price:.2f} | "
              f"Change: {change_pct:.2f}%")
        
        return {
            'sentiment': sentiment,
            'headlines': headlines[:5],
            'intraday_data': intraday_data,
            'forecast': forecast_series,
            'start_price': start_price,
            'end_price': end_price,
            'backtest': backtest_results,
            'accuracy': backtest_accuracy,
            'trading_metrics': trading_metrics,
            'equity_curve': equity_curve,
            'forecast_bars': forecast_bars
        }
    except Exception as e:
        print(f"  ❌ Forecasting failed: {str(e)}")
        traceback.print_exc()
        return None


def generate_intraday_report(analysis_results, interval):
    if not analysis_results:
        print("❌ No analysis results to report")
        return
        
    print(f"\n📊 CRUDE OIL INTRADAY ANALYSIS REPORT ({interval} BARS)")
    print("=" * 70)
    print(f"- Strategy Horizon: {STRATEGY_HORIZON}")
    print(f"- News Sentiment: {analysis_results['sentiment']:.2f}")
    
    if analysis_results.get('accuracy') is not None:
        print(f"- Backtest Accuracy: {analysis_results['accuracy']:.1f}%")
    
    if 'start_price' in analysis_results and 'end_price' in analysis_results:
        price_change = analysis_results['end_price'] - analysis_results['start_price']
        trend = "↑ BULLISH" if price_change > 0 else "↓ BEARISH"
        print(f"\n- Forecast Trend: {trend} (${price_change:.2f} change)")
    
    if analysis_results.get('trading_metrics'):
        metrics = analysis_results['trading_metrics']
        print(f"\n📊 TRADING PERFORMANCE ({STRATEGY_HORIZON} horizon):")
        print(f"  - Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"  - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  - Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} min")
        print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    print("\n- Key News Headlines:")
    for i, headline in enumerate(analysis_results['headlines'][:3], 1):
        print(f"  {i}. {headline}")
    
    print("\n✅ Report generated successfully")
    print("=" * 70)


def create_pdf_report(analysis_results, interval):
    if not PDF_AVAILABLE or not analysis_results:
        print("⚠️ PDF creation disabled or no results available")
        return None
    
    # Function to clean text for PDF compatibility
    def clean_text(text):
        """Replace problematic Unicode characters with ASCII equivalents"""
        if not isinstance(text, str):
            return str(text)
        replacements = {
            '\u2014': '-',  # Em dash
            '\u2013': '-',  # En dash
            '\u2018': "'",  # Left single quotation
            '\u2019': "'",  # Right single quotation
            '\u201c': '"',  # Left double quotation
            '\u201d': '"',  # Right double quotation
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',   # Non-breaking space
            '\u00ae': '(R)', # Registered trademark
            '\u00a9': '(C)', # Copyright
            '\u2122': '(TM)' # Trademark
        }
        for uni_char, ascii_sub in replacements.items():
            text = text.replace(uni_char, ascii_sub)
        return text
    
    forecast_bars = analysis_results.get('forecast_bars', 12)
    
    temp_dir = tempfile.mkdtemp()
    forecast_img = os.path.join(temp_dir, 'oil_forecast.png')
    equity_img = os.path.join(temp_dir, 'intraday_portfolio_equity.png')
    
    if 'intraday_data' in analysis_results and 'forecast' in analysis_results:
        plt.figure(figsize=(12, 6))
        plt.plot(analysis_results['intraday_data'].index, analysis_results['intraday_data']['price'], label='Historical Prices', color='blue')
        
        if isinstance(analysis_results['forecast'], pd.Series):
            plt.plot(analysis_results['forecast'].index, analysis_results['forecast'], label='Forecast', color='red', linestyle='--')
        else:
            plt.scatter(analysis_results['intraday_data'].index[-1] + pd.Timedelta(minutes=30), 
                        analysis_results['forecast'], 
                        color='red', marker='o', s=100, label='Forecast')
        
        plt.title(f"Crude Oil {interval} Price Forecast ({STRATEGY_HORIZON} horizon)")
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig(forecast_img)
        plt.close()
    
    if 'equity_curve' in analysis_results:
        plt.figure(figsize=(12, 6))
        plt.plot(analysis_results['equity_curve'], label='Portfolio Value')
        plt.title(f"Intraday Portfolio Equity Curve ({STRATEGY_HORIZON} horizon)")
        plt.ylabel("Value ($)")
        plt.xlabel("Trade")
        plt.legend()
        plt.grid(True)
        plt.savefig(equity_img)
        plt.close()
    
    pdf = FPDF_CLASS()
    pdf.add_page()
    
    # Set safe margins
    pdf.set_margins(15, 15, 15)
    effective_width = 210 - 30  # Page width minus left+right margins
    
    # Add DejaVu font if available (Unicode support)
    dejavu_path = None
    font_name = "helvetica"
    try:
        # Check common locations for DejaVu font
        possible_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/DejaVuSans.ttf",
            "C:/Windows/Fonts/DejaVuSans.ttf",
            os.path.expanduser("~/.fonts/DejaVuSans.ttf")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dejavu_path = path
                pdf.add_font('DejaVu', '', dejavu_path, uni=True)
                font_name = "DejaVu"
                print(f"✅ Using Unicode font: {dejavu_path}")
                break
    except Exception as e:
        print(f"⚠️ Font setup error: {e}. Using core font helvetica.")
    
    # Set font based on availability
    pdf.set_font(font_name, size=10)
    
    # Title section
    pdf.set_font(font_name, 'B', 16)
    title = clean_text(f"Crude Oil Intraday Analysis ({interval} bars)")
    pdf.cell(effective_width, 10, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font(font_name, size=10)
    gen_date = clean_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.cell(effective_width, 8, text=gen_date, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(8)
    
    # Executive Summary
    pdf.set_font(font_name, 'B', 14)
    pdf.cell(effective_width, 8, text=clean_text("Executive Summary"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)
    
    # Add summary text with proper wrapping
    summary_text = ""
    if 'start_price' in analysis_results and 'end_price' in analysis_results:
        price_change = analysis_results['end_price'] - analysis_results['start_price']
        trend = "BULLISH" if price_change > 0 else "BEARISH"
        summary_text += clean_text(f"Forecast Trend: {trend} (${price_change:.2f} change over {forecast_bars} bars)\n")
    
    sentiment = analysis_results.get('sentiment', 0)
    sentiment_label = "Positive" if sentiment > 0 else ("Negative" if sentiment < 0 else "Neutral")
    summary_text += clean_text(f"News Sentiment: {sentiment_label} ({sentiment:.2f})\n")
    
    if analysis_results.get('trading_metrics'):
        metrics = analysis_results['trading_metrics']
        summary_text += clean_text(f"Simulated Trading Return: {metrics.get('total_return', 0):.2f}%\n")
        summary_text += clean_text(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n")
    
    pdf.multi_cell(effective_width, 6, text=clean_text(summary_text))
    pdf.ln(8)
    
    # Images section
    img_height = 60
    if os.path.exists(forecast_img):
        pdf.set_font(font_name, 'B', 12)
        pdf.cell(effective_width, 8, text=clean_text("Price Forecast:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(forecast_img, x=15, w=effective_width, h=img_height)
        pdf.ln(5)
    
    if os.path.exists(equity_img):
        pdf.set_font(font_name, 'B', 12)
        pdf.cell(effective_width, 8, text=clean_text("Portfolio Performance:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(equity_img, x=15, w=effective_width, h=img_height)
        pdf.ln(5)
    
    # Headlines section
    pdf.set_font(font_name, 'B', 12)
    pdf.cell(effective_width, 8, text=clean_text("Key News Headlines:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)
    
    cleaned_headlines = [clean_text(h) for h in analysis_results.get('headlines', [])[:3]]
    for headline in cleaned_headlines:
        # Handle long headlines by splitting into words
        words = headline.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if pdf.get_string_width(test_line) < effective_width - 5:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
            
        for line in lines:
            pdf.cell(effective_width, 6, text=line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
    
    # Recommendations section
    pdf.set_font(font_name, 'B', 12)
    pdf.cell(effective_width, 8, text=clean_text("Intraday Trading Recommendations:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)
    
    recommendations = [
        "Monitor volume spikes at key technical levels",
        "Watch for economic releases (EIA reports at 10:30 AM EST)",
        "Close positions before market close to avoid overnight risk",
        "Use tight stop-losses for intraday positions",
        "Focus on high-probability setups during market open (9:30-11 AM EST)"
    ]
    cleaned_recs = [clean_text(rec) for rec in recommendations]
    for rec in cleaned_recs:
        pdf.cell(effective_width, 6, text=f"- {rec}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Footer
    pdf.ln(10)
    pdf.set_font(font_name, 'I', 8)
    footer = clean_text(f"Generated by Crude Oil Intraday Analyzer | Strategy Horizon: {STRATEGY_HORIZON}")
    pdf.cell(effective_width, 5, text=footer, align='C')
    
    pdf_file = f"CrudeOil_Intraday_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(pdf_file)
    
    print(f"✅ PDF report generated: {pdf_file}")
    return pdf_file


def email_report(pdf_file, recipient):
    try:
        sender_email = "your_email@example.com"
        password = "your_email_password"
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = f'Crude Oil Intraday Report - {datetime.now().strftime("%Y-%m-%d")}'
        
        body = f"Attached is the Crude Oil Intraday Analysis Report for {datetime.now().strftime('%Y-%m-%d')}"
        msg.attach(MIMEText(body, 'plain'))
        
        with open(pdf_file, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_file))
            msg.attach(attach)
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Report emailed to {recipient}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {str(e)}")
        return False


GOOGLE_CLIENT_ID = "207006177171-63vrp2gqi11h47k87kocrka7hfhtgjul.apps.googleusercontent.com"

def install_gcloud_storage():
    print("⚠️ Google Cloud Storage library not found. Attempting to install...")
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "google-cloud-storage"
        ])
        print("✅ Google Cloud Storage installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install Google Cloud Storage: {str(e)}")
        print("  Please install manually with: pip install google-cloud-storage")
        return False

def upload_to_gcs(pdf_file, bucket_name):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        if not install_gcloud_storage():
            return False
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
        except ImportError:
            print("❌ Still unable to import google.cloud.storage after installation")
            return False
    
    try:
        credentials = service_account.Credentials.from_service_account_info(
            {"type": "service_account", "client_id": GOOGLE_CLIENT_ID}
        )
        
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"reports/{os.path.basename(pdf_file)}")
        blob.upload_from_filename(pdf_file)
        
        print(f"✅ Report uploaded to GCS using client ID: {GOOGLE_CLIENT_ID}")
        return True
    except Exception as e:
        print(f"❌ GCS upload failed: {str(e)}")
        return False


if __name__ == '__main__':
    print(f"Python version: {sys.version}")
    print(f"Running from: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Crude Oil Intraday Analysis')
    parser.add_argument('--interval', type=str, default='30m', choices=VALID_INTERVALS,
                        help=f'Intraday time interval (default: 30m). Valid: {VALID_INTERVALS}')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data (default: 30)')
    parser.add_argument('--email', type=str, default=None,
                        help='Email address to send report')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                        help='Google Cloud Storage bucket name for report upload')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Crude Oil Analysis - Strategy Horizon: {STRATEGY_HORIZON}")
    print(f"  - Interval: {args.interval}")
    print(f"  - Days: {args.days}")
    if args.email: print(f"  - Email report to: {args.email}")
    if args.gcs_bucket: print(f"  - Upload report to GCS bucket: {args.gcs_bucket}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    analysis_results = analyze_crude_oil_intraday(
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        forecast_bars=12,
        backtest=True
    )
    
    if analysis_results:
        generate_intraday_report(analysis_results, args.interval)
        
        # We always have matplotlib since it's required for plotting
        pdf_file = create_pdf_report(analysis_results, args.interval)
        if pdf_file:
            if args.email:
                email_report(pdf_file, args.email)
            if args.gcs_bucket:
                upload_to_gcs(pdf_file, args.gcs_bucket)
    
    print("\n🏁 Analysis complete!")
