"""Tests for the pure technical-indicator functions."""
import numpy as np
import pandas as pd

from crude_oil_analysis.indicators import calculate_atr, calculate_macd, calculate_rsi


def _frame(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="30min")
    df = pd.DataFrame({"price": prices}, index=idx)
    df["high"] = df["price"] + 0.5
    df["low"] = df["price"] - 0.5
    return df


def test_rsi_within_bounds():
    df = _frame(np.linspace(70, 80, 50))
    rsi = calculate_rsi(df)
    assert len(rsi) == len(df)
    assert rsi.between(0, 100).all()


def test_rsi_neutral_for_single_row():
    df = _frame([75.0])
    assert calculate_rsi(df).tolist() == [50]


def test_macd_zero_when_too_short():
    df = _frame([75.0, 75.1, 75.2])  # fewer than `slow` (26) rows
    macd, signal = calculate_macd(df)
    assert (macd == 0).all() and (signal == 0).all()


def test_macd_shapes_align():
    df = _frame(np.linspace(70, 80, 60))
    macd, signal = calculate_macd(df)
    assert len(macd) == len(df) == len(signal)


def test_atr_non_negative():
    df = _frame(np.linspace(70, 80, 40))
    atr = calculate_atr(df)
    assert (atr >= 0).all()
