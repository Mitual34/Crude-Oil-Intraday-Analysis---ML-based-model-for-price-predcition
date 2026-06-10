"""Crude Oil Intraday Analysis.

A machine-learning toolkit that forecasts intraday crude-oil (WTI) prices using
XGBoost, technical indicators and news sentiment, with a realistic backtester
and automated PDF/email/cloud reporting.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crude-oil-analysis")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
