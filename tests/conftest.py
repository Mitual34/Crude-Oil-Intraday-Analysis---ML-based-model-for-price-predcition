"""Shared test configuration."""
import os

# Use a non-interactive matplotlib backend so plotting never needs a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def intraday_data():
    """A deterministic synthetic intraday OHLCV frame with a DatetimeIndex."""
    n = 200
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="30min")
    rng = np.random.default_rng(0)
    price = 75 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": price - 0.1,
            "high": price + 0.2,
            "low": price - 0.2,
            "price": price,
            "volume": rng.integers(10000, 50000, n),
            "usd_index": rng.uniform(90, 110, n),
        },
        index=idx,
    )
