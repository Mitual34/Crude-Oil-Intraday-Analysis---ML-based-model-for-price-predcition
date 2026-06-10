"""Pure technical-indicator functions.

These operate on a DataFrame with a ``price`` (and, for ATR, ``high``/``low``)
column and return a pandas Series aligned to the input index. They hold no
state, which makes them straightforward to unit-test without market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index. Returns a neutral 50 when data is too short."""
    if len(data) < 2:
        return pd.Series([50] * len(data), index=data.index)

    delta = data["price"].diff().fillna(0)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean().fillna(0)
    avg_loss = loss.rolling(window=window, min_periods=1).mean().fillna(0)

    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line. Returns zeros when data is shorter than ``slow``."""
    if len(data) < slow:
        return pd.Series(0, index=data.index), pd.Series(0, index=data.index)

    exp1 = data["price"].ewm(span=fast, adjust=False).mean()
    exp2 = data["price"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.fillna(0), signal_line.fillna(0)


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range. Returns zeros when data is too short."""
    if len(data) < 2:
        return pd.Series(0, index=data.index)

    prev_close = data["price"].shift().bfill()
    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - prev_close)
    low_close = np.abs(data["low"] - prev_close)

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()
    return atr.fillna(0)
