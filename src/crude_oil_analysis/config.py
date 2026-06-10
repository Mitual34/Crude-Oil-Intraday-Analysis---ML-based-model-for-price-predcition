"""Static configuration and environment-driven settings.

Secrets (email credentials, Google client id) are read from environment
variables rather than hard-coded, so they never live in source control.
"""
from __future__ import annotations

import os
from datetime import timedelta

# --- Strategy horizon -------------------------------------------------------
STRATEGY_HORIZON = "intraday"
VALID_INTERVALS = ["5m", "15m", "30m", "60m"]
MAX_HOLD_PERIOD = timedelta(minutes=5)

# Features the model expects, in a stable order.
EXPECTED_FEATURES = [
    "rsi", "macd", "macd_signal", "atr", "volume_ma", "volume_pct_change",
    "returns", "log_returns", "hour", "minute", "day_of_week", "time_of_day",
    "economic_event", "news_sentiment",
    "returns_lag1", "returns_lag2", "returns_lag3", "returns_lag5", "returns_lag8",
]

# --- Email (SMTP) settings, configured via environment ----------------------
SENDER_EMAIL = os.environ.get("OIL_REPORT_SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("OIL_REPORT_SENDER_PASSWORD", "")
SMTP_SERVER = os.environ.get("OIL_REPORT_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("OIL_REPORT_SMTP_PORT", "587"))

# --- Google Cloud Storage upload, configured via environment ----------------
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
