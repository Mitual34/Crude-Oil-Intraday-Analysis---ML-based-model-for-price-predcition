# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-10

### Added
- Installable `crude_oil_analysis` package with a `crude-oil-analysis` console
  command and `python -m crude_oil_analysis` entry point.
- pytest suite covering indicators, sentiment, portfolio metrics, the CLI, and a
  network-free feature → train → forecast pipeline test.
- GitHub Actions CI across Python 3.9–3.12.
- `requirements.txt`, `pyproject.toml`, `.gitignore`, `.gitattributes`,
  `.env.example` and a documented README.

### Changed
- Restructured the original single-file `Crude Oil Analysis.py` script into a
  logically separated package (config, optional_deps, indicators, news,
  portfolio, forecaster, backtester, analysis, reporting, cli).
- Technical indicators (RSI/MACD/ATR) extracted into pure, testable functions.

### Security / Robustness
- Removed runtime `pip install` calls at import time in favour of clean optional
  imports with graceful fallbacks.
- Moved email and Google Cloud credentials out of source code into environment
  variables (see `.env.example`).
- Forced UTF-8 stdout/stderr so the emoji-rich output no longer crashes on
  Windows consoles (cp1252 `UnicodeEncodeError`).
- Dropped unused imports (bs4, dateparser, requests, scipy, statsmodels, …),
  trimming the dependency footprint.
