"""End-to-end (network-free) test of the feature -> train -> forecast pipeline."""
import numpy as np
import pandas as pd

from crude_oil_analysis.config import EXPECTED_FEATURES
from crude_oil_analysis.forecaster import CommodityForecaster


def test_create_features_produces_clean_expected_columns(intraday_data):
    f = CommodityForecaster(interval="30m", forecast_bars=5)
    feats = f.create_features(intraday_data.copy())

    for col in EXPECTED_FEATURES:
        assert col in feats.columns, f"missing feature: {col}"

    # No infinities or NaNs should survive feature engineering.
    assert not np.isinf(feats.to_numpy()).any()
    assert not feats.isnull().to_numpy().any()


def test_train_then_forecast_returns_series(intraday_data):
    f = CommodityForecaster(interval="30m", forecast_bars=6)
    feats = f.create_features(intraday_data.copy())
    X = feats.drop("price", axis=1)
    y = feats["price"]

    f.train_model(X, y)
    assert f.feature_columns is not None

    forecast = f.forecast_prices(intraday_data, news_sentiment=0.1)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == 6
    assert forecast.notna().all()


def test_invalid_interval_raises():
    import pytest
    with pytest.raises(ValueError):
        CommodityForecaster(interval="1d")
