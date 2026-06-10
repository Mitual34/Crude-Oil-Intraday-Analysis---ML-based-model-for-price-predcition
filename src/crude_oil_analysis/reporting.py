"""Reporting: console summary, PDF report, email delivery and GCS upload."""
from __future__ import annotations

import os
import smtplib
import tempfile
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    GOOGLE_CLIENT_ID,
    SENDER_EMAIL,
    SENDER_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
    STRATEGY_HORIZON,
)
from .optional_deps import FPDF_CLASS, PDF_AVAILABLE, XPos, YPos


def generate_intraday_report(analysis_results, interval):
    if not analysis_results:
        print("❌ No analysis results to report")
        return

    print(f"\n📊 CRUDE OIL INTRADAY ANALYSIS REPORT ({interval} BARS)")
    print("=" * 70)
    print(f"- Strategy Horizon: {STRATEGY_HORIZON}")
    print(f"- News Sentiment: {analysis_results['sentiment']:.2f}")

    if analysis_results.get('accuracy') is not None:
        print(f"- Backtest Accuracy: {analysis_results['accuracy']:.1f}%")

    if 'start_price' in analysis_results and 'end_price' in analysis_results:
        price_change = analysis_results['end_price'] - analysis_results['start_price']
        trend = "↑ BULLISH" if price_change > 0 else "↓ BEARISH"
        print(f"\n- Forecast Trend: {trend} (${price_change:.2f} change)")

    if analysis_results.get('trading_metrics'):
        metrics = analysis_results['trading_metrics']
        print(f"\n📊 TRADING PERFORMANCE ({STRATEGY_HORIZON} horizon):")
        print(f"  - Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"  - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  - Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} min")
        print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

    print("\n- Key News Headlines:")
    for i, headline in enumerate(analysis_results['headlines'][:3], 1):
        print(f"  {i}. {headline}")

    print("\n✅ Report generated successfully")
    print("=" * 70)


def _clean_text(text):
    """Replace problematic Unicode characters with ASCII equivalents for PDF."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        '—': '-', '–': '-', '‘': "'", '’': "'",
        '“': '"', '”': '"', '…': '...', ' ': ' ',
        '®': '(R)', '©': '(C)', '™': '(TM)',
    }
    for uni_char, ascii_sub in replacements.items():
        text = text.replace(uni_char, ascii_sub)
    return text


def create_pdf_report(analysis_results, interval):
    if not PDF_AVAILABLE or not analysis_results:
        print("⚠️ PDF creation disabled or no results available")
        return None

    forecast_bars = analysis_results.get('forecast_bars', 12)

    temp_dir = tempfile.mkdtemp()
    forecast_img = os.path.join(temp_dir, 'oil_forecast.png')
    equity_img = os.path.join(temp_dir, 'intraday_portfolio_equity.png')

    if 'intraday_data' in analysis_results and 'forecast' in analysis_results:
        plt.figure(figsize=(12, 6))
        plt.plot(analysis_results['intraday_data'].index, analysis_results['intraday_data']['price'], label='Historical Prices', color='blue')

        if isinstance(analysis_results['forecast'], pd.Series):
            plt.plot(analysis_results['forecast'].index, analysis_results['forecast'], label='Forecast', color='red', linestyle='--')
        else:
            plt.scatter(analysis_results['intraday_data'].index[-1] + pd.Timedelta(minutes=30),
                        analysis_results['forecast'],
                        color='red', marker='o', s=100, label='Forecast')

        plt.title(f"Crude Oil {interval} Price Forecast ({STRATEGY_HORIZON} horizon)")
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig(forecast_img)
        plt.close()

    if 'equity_curve' in analysis_results:
        plt.figure(figsize=(12, 6))
        plt.plot(analysis_results['equity_curve'], label='Portfolio Value')
        plt.title(f"Intraday Portfolio Equity Curve ({STRATEGY_HORIZON} horizon)")
        plt.ylabel("Value ($)")
        plt.xlabel("Trade")
        plt.legend()
        plt.grid(True)
        plt.savefig(equity_img)
        plt.close()

    pdf = FPDF_CLASS()
    pdf.add_page()

    # Set safe margins
    pdf.set_margins(15, 15, 15)
    effective_width = 210 - 30  # Page width minus left+right margins

    # Add DejaVu font if available (Unicode support)
    font_name = "helvetica"
    try:
        possible_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/DejaVuSans.ttf",
            "C:/Windows/Fonts/DejaVuSans.ttf",
            os.path.expanduser("~/.fonts/DejaVuSans.ttf"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pdf.add_font('DejaVu', '', path, uni=True)
                font_name = "DejaVu"
                print(f"✅ Using Unicode font: {path}")
                break
    except Exception as e:
        print(f"⚠️ Font setup error: {e}. Using core font helvetica.")

    pdf.set_font(font_name, size=10)

    # Title section
    pdf.set_font(font_name, 'B', 16)
    title = _clean_text(f"Crude Oil Intraday Analysis ({interval} bars)")
    pdf.cell(effective_width, 10, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.set_font(font_name, size=10)
    gen_date = _clean_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.cell(effective_width, 8, text=gen_date, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(8)

    # Executive Summary
    pdf.set_font(font_name, 'B', 14)
    pdf.cell(effective_width, 8, text=_clean_text("Executive Summary"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)

    summary_text = ""
    if 'start_price' in analysis_results and 'end_price' in analysis_results:
        price_change = analysis_results['end_price'] - analysis_results['start_price']
        trend = "BULLISH" if price_change > 0 else "BEARISH"
        summary_text += _clean_text(f"Forecast Trend: {trend} (${price_change:.2f} change over {forecast_bars} bars)\n")

    sentiment = analysis_results.get('sentiment', 0)
    sentiment_label = "Positive" if sentiment > 0 else ("Negative" if sentiment < 0 else "Neutral")
    summary_text += _clean_text(f"News Sentiment: {sentiment_label} ({sentiment:.2f})\n")

    if analysis_results.get('trading_metrics'):
        metrics = analysis_results['trading_metrics']
        summary_text += _clean_text(f"Simulated Trading Return: {metrics.get('total_return', 0):.2f}%\n")
        summary_text += _clean_text(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n")

    pdf.multi_cell(effective_width, 6, text=_clean_text(summary_text))
    pdf.ln(8)

    # Images section
    img_height = 60
    if os.path.exists(forecast_img):
        pdf.set_font(font_name, 'B', 12)
        pdf.cell(effective_width, 8, text=_clean_text("Price Forecast:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(forecast_img, x=15, w=effective_width, h=img_height)
        pdf.ln(5)

    if os.path.exists(equity_img):
        pdf.set_font(font_name, 'B', 12)
        pdf.cell(effective_width, 8, text=_clean_text("Portfolio Performance:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(equity_img, x=15, w=effective_width, h=img_height)
        pdf.ln(5)

    # Headlines section
    pdf.set_font(font_name, 'B', 12)
    pdf.cell(effective_width, 8, text=_clean_text("Key News Headlines:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)

    cleaned_headlines = [_clean_text(h) for h in analysis_results.get('headlines', [])[:3]]
    for headline in cleaned_headlines:
        words = headline.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            if pdf.get_string_width(test_line) < effective_width - 5:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines:
            pdf.cell(effective_width, 6, text=line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # Recommendations section
    pdf.set_font(font_name, 'B', 12)
    pdf.cell(effective_width, 8, text=_clean_text("Intraday Trading Recommendations:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, size=10)

    recommendations = [
        "Monitor volume spikes at key technical levels",
        "Watch for economic releases (EIA reports at 10:30 AM EST)",
        "Close positions before market close to avoid overnight risk",
        "Use tight stop-losses for intraday positions",
        "Focus on high-probability setups during market open (9:30-11 AM EST)",
    ]
    cleaned_recs = [_clean_text(rec) for rec in recommendations]
    for rec in cleaned_recs:
        pdf.cell(effective_width, 6, text=f"- {rec}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Footer
    pdf.ln(10)
    pdf.set_font(font_name, 'I', 8)
    footer = _clean_text(f"Generated by Crude Oil Intraday Analyzer | Strategy Horizon: {STRATEGY_HORIZON}")
    pdf.cell(effective_width, 5, text=footer, align='C')

    pdf_file = f"CrudeOil_Intraday_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(pdf_file)

    print(f"✅ PDF report generated: {pdf_file}")
    return pdf_file


def email_report(pdf_file, recipient):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ Email not configured. Set OIL_REPORT_SENDER_EMAIL and "
              "OIL_REPORT_SENDER_PASSWORD environment variables.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        msg['Subject'] = f'Crude Oil Intraday Report - {datetime.now().strftime("%Y-%m-%d")}'

        body = f"Attached is the Crude Oil Intraday Analysis Report for {datetime.now().strftime('%Y-%m-%d')}"
        msg.attach(MIMEText(body, 'plain'))

        with open(pdf_file, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_file))
            msg.attach(attach)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"✅ Report emailed to {recipient}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {str(e)}")
        return False


def upload_to_gcs(pdf_file, bucket_name):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        print("❌ google-cloud-storage not installed. "
              "Install the optional extra with: pip install 'crude-oil-analysis[gcs]'")
        return False

    try:
        credentials = service_account.Credentials.from_service_account_info(
            {"type": "service_account", "client_id": GOOGLE_CLIENT_ID}
        )

        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"reports/{os.path.basename(pdf_file)}")
        blob.upload_from_filename(pdf_file)

        print(f"✅ Report uploaded to GCS bucket '{bucket_name}'")
        return True
    except Exception as e:
        print(f"❌ GCS upload failed: {str(e)}")
        return False
