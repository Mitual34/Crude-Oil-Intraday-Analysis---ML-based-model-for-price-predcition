"""High-level analysis pipeline: news -> data -> backtest -> forecast."""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta

from .backtester import IntradayBacktester
from .config import STRATEGY_HORIZON
from .forecaster import CommodityForecaster
from .news import GoogleNews, analyze_sentiment


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
