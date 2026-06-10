# Crude Oil Intraday Analysis

[![CI](https://github.com/Mitual34/Crude-Oil-Intraday-Analysis---ML-based-model-for-price-predcition/actions/workflows/ci.yml/badge.svg)](https://github.com/Mitual34/Crude-Oil-Intraday-Analysis---ML-based-model-for-price-predcition/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: XGBoost](https://img.shields.io/badge/model-XGBoost-brightgreen.svg)](https://xgboost.readthedocs.io/)

A machine-learning toolkit for **intraday crude-oil (WTI) price forecasting**. It
combines an XGBoost model, classic technical indicators, and real-time news
sentiment, then evaluates a trading strategy through a realistic walk-forward
backtester with spread, slippage and volatility-adjusted position sizing —
finishing with an automated PDF report that can be emailed or pushed to Google
Cloud Storage.

> ⚠️ **Disclaimer — read first.** This is a research/educational project, **not
> financial advice** and not a production trading system. Backtested figures are
> highly sensitive to the data window and assumptions; the annualised Sharpe
> ratio in particular is inflated by intraday annualisation and is *not* a
> realistic estimate of live performance. Markets involve substantial risk of
> loss. Do not trade real capital based on this code.

---

## ✨ Features

- **XGBoost forecasting** of the next intraday bars (auto-falls back to a
  RandomForest if XGBoost is unavailable).
- **Technical indicators**: RSI, MACD, ATR, volume dynamics, returns and lagged
  returns, plus time-of-day / economic-event flags (market open, EIA report).
- **News sentiment** from Google News headlines scored with TextBlob.
- **Walk-forward backtester** modelling spread, slippage, a max-hold period and
  volatility-based position sizing, reporting return, win rate, Sharpe/Sortino,
  drawdown and profit factor.
- **Automated reporting**: console summary, a polished PDF (charts + headlines +
  recommendations), optional email delivery and optional Google Cloud upload.
- **Live market data** via `yfinance` (`CL=F`), with a synthetic fallback so the
  pipeline still runs when data is unavailable.

## 🏗️ Architecture

```
            news (Google News + TextBlob)
                        │  sentiment
                        ▼
 yfinance ─▶ forecaster ─▶ create_features ─▶ XGBoost ─▶ forecast
   (CL=F)        │              (RSI/MACD/ATR/…)             │
                 ▼                                           ▼
           backtester ──▶ portfolio (spread/slippage) ──▶ metrics
                                                            │
                                              reporting (console / PDF / email / GCS)
```

## 📦 Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/Mitual34/Crude-Oil-Intraday-Analysis---ML-based-model-for-price-predcition.git
cd Crude-Oil-Intraday-Analysis---ML-based-model-for-price-predcition

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install as a package (provides the `crude-oil-analysis` command)
pip install .

# ...or just the dependencies
pip install -r requirements.txt

# Optional: enable Google Cloud Storage upload
pip install ".[gcs]"
```

## 🚀 Usage

```bash
# Default: 30-minute bars, last 30 days
crude-oil-analysis

# Choose interval and history window
crude-oil-analysis --interval 15m --days 45

# Email the PDF report (requires the env vars below)
crude-oil-analysis --email you@example.com

# Upload the PDF to a GCS bucket
crude-oil-analysis --gcs-bucket my-reports-bucket
```

Without installing the package you can also run `python -m crude_oil_analysis ...`.

| Flag | Default | Description |
|------|---------|-------------|
| `--interval` | `30m` | Bar size: `5m`, `15m`, `30m`, `60m` |
| `--days` | `30` | Days of historical data to pull |
| `--email` | – | Recipient address for the PDF report |
| `--gcs-bucket` | – | GCS bucket to upload the PDF to |

## 🔐 Configuration (secrets via environment)

Email and Google Cloud credentials are read from environment variables — they
are **never** hard-coded. Copy [`.env.example`](.env.example) to `.env` and fill
it in (or export the variables in your shell):

| Variable | Used for |
|----------|----------|
| `OIL_REPORT_SENDER_EMAIL` / `OIL_REPORT_SENDER_PASSWORD` | SMTP login for `--email` |
| `OIL_REPORT_SMTP_SERVER` / `OIL_REPORT_SMTP_PORT` | SMTP host/port (default Gmail) |
| `GOOGLE_CLIENT_ID` | Google Cloud identity for `--gcs-bucket` |

## 📤 Outputs

Each run writes to the working directory (all git-ignored):

- `oil_forecast.png` — historical prices + forecast
- `backtest_results.png` — actual vs forecast with trade markers
- `intraday_portfolio_equity.png` — equity curve
- `CrudeOil_Intraday_Report_<timestamp>.pdf` — the full report

## 🗂️ Project structure

```
src/crude_oil_analysis/
├── config.py          # constants + env-driven settings (secrets)
├── optional_deps.py   # graceful optional imports (xgboost, fpdf2, feedparser)
├── indicators.py      # pure RSI / MACD / ATR functions
├── news.py            # Google News client + sentiment
├── portfolio.py       # trade accounting & performance metrics
├── forecaster.py      # data, features, model training & forecasting
├── backtester.py      # walk-forward intraday backtest
├── analysis.py        # end-to-end pipeline orchestration
├── reporting.py       # console / PDF / email / GCS reporting
└── cli.py             # argparse entry point
tests/                 # pytest suite (network-free)
.github/workflows/     # CI across Python 3.9–3.12
```

## ✅ Development & testing

```bash
pip install -e ".[dev]"
pytest
```

The suite is fully offline — it exercises the indicators, sentiment, portfolio
metrics, CLI parsing, and a complete feature → train → forecast pipeline on
deterministic synthetic data (live data fetching is not exercised in tests).

## 📄 License

Released under the [MIT License](LICENSE).
