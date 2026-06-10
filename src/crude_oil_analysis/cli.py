"""Command-line interface for the crude-oil intraday analyzer."""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

from .analysis import analyze_crude_oil_intraday
from .config import STRATEGY_HORIZON, VALID_INTERVALS
from .reporting import create_pdf_report, email_report, generate_intraday_report, upload_to_gcs

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _force_utf8_output():
    """Make stdout/stderr UTF-8 so emoji in messages never crash on Windows (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Crude Oil Intraday Analysis')
    parser.add_argument('--interval', type=str, default='30m', choices=VALID_INTERVALS,
                        help=f'Intraday time interval (default: 30m). Valid: {VALID_INTERVALS}')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data (default: 30)')
    parser.add_argument('--email', type=str, default=None,
                        help='Email address to send the report (requires OIL_REPORT_SENDER_* env vars)')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                        help='Google Cloud Storage bucket name for report upload')
    return parser


def main(argv=None):
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Python version: {sys.version}")
    print(f"Running from: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

    print(f"🚀 Starting Crude Oil Analysis - Strategy Horizon: {STRATEGY_HORIZON}")
    print(f"  - Interval: {args.interval}")
    print(f"  - Days: {args.days}")
    if args.email:
        print(f"  - Email report to: {args.email}")
    if args.gcs_bucket:
        print(f"  - Upload report to GCS bucket: {args.gcs_bucket}")

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

        pdf_file = create_pdf_report(analysis_results, args.interval)
        if pdf_file:
            if args.email:
                email_report(pdf_file, args.email)
            if args.gcs_bucket:
                upload_to_gcs(pdf_file, args.gcs_bucket)

    print("\n🏁 Analysis complete!")


if __name__ == '__main__':
    main()
