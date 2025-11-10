import os
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# ==================== Config ====================

TABLE_NAME = os.environ.get("TABLE_NAME", "FinancialAnomalies")
TOPIC_ARN = os.environ.get("TOPIC_ARN")
S3_BUCKET = os.environ.get("S3_BUCKET", "financial-anomaly-reports")

TICKER = os.environ.get("TICKER", "AAPL")
FETCH_PERIOD = os.environ.get("FETCH_PERIOD", "2d")
FETCH_INTERVAL = os.environ.get("FETCH_INTERVAL", "1m")
ROLLING_WINDOW = int(os.environ.get("ROLLING_WINDOW", "20"))
CONTAMINATION = float(os.environ.get("CONTAMINATION", "0.02"))

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
sns = boto3.client("sns")
s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ==================== Data stuff ====================

def fetch_data(ticker=TICKER, period=FETCH_PERIOD, interval=FETCH_INTERVAL):
    """Grab price data from Yahoo Finance"""
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        
        if df is None or df.empty:
            logger.warning(f"Got nothing back for {ticker}")
            return pd.DataFrame()
        
        df = df.reset_index()
        
        # Yahoo returns different column names depending on interval
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        logger.info(f"Grabbed {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Couldn't fetch data for {ticker}: {str(e)}")
        return pd.DataFrame()


def add_features(df):
    """Calculate returns, gaps, volatility - the usual suspects"""
    df['return'] = df['Close'].pct_change()
    df['hl_gap'] = df['High'] - df['Low']
    df['oc_gap'] = df['Open'] - df['Close']
    df['volatility'] = df['return'].rolling(window=ROLLING_WINDOW, min_periods=1).std()
    
    df.dropna(inplace=True)
    return df


# ==================== ML detection ====================

def standardize(df_window, features):
    """Scale features to mean=0, std=1 so they're comparable"""
    scaler = StandardScaler()
    X = df_window[features].values
    return scaler.fit_transform(X)


def get_anomaly_scores(X_window):
    """
    Run two different anomaly detectors and combine them.
    IsolationForest is good for big outliers.
    LOF catches local weirdness.
    """
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42)
    iso.fit(X_window)
    iso_scores = -iso.score_samples(X_window)
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=False)
    lof_labels = lof.fit_predict(X_window)
    lof_scores = (lof_labels == -1).astype(float)
    
    # Normalize ISO scores to 0-1 range
    iso_min, iso_max = np.min(iso_scores), np.max(iso_scores)
    iso_norm = (iso_scores - iso_min) / (iso_max - iso_min + 1e-9)
    
    # Weight ISO more heavily (60/40 split)
    combined = 0.6 * iso_norm + 0.4 * lof_scores
    
    return combined


def check_for_anomaly(df, features):
    """Look at the most recent data point and see if it's weird"""
    if len(df) < ROLLING_WINDOW + 1:
        logger.warning(f"Not enough data: {len(df)} points, need at least {ROLLING_WINDOW + 1}")
        return None
    
    # Grab a window including the latest point
    window_df = df.iloc[-(ROLLING_WINDOW + 1):].copy().reset_index(drop=True)
    
    X_scaled = standardize(window_df, features)
    scores = get_anomaly_scores(X_scaled)
    
    latest_score = float(scores[-1])
    threshold = np.percentile(scores, 100 * (1 - CONTAMINATION))
    
    if latest_score < threshold:
        logger.info(f"Score {latest_score:.3f} is below threshold {threshold:.3f}, all good")
        return None
    
    logger.warning(f"Found anomaly! Score: {latest_score:.3f} vs threshold {threshold:.3f}")
    
    latest_row = window_df.iloc[-1].to_dict()
    
    return {
        "score": latest_score,
        "threshold": float(threshold),
        "latest": latest_row,
        "window_scores": scores.tolist()
    }


# ==================== Reporting ====================

def make_report(ticker, anomaly, df, summary):
    """Generate a PDF with charts and details"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_name = f"{ticker}_anomaly_{timestamp}.pdf"
    report_path = os.path.join("/tmp", report_name)
    
    try:
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title and header info
        story.append(Paragraph(f"<b>Anomaly Alert: {ticker}</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", 
            styles["Normal"]
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Score:</b> {anomaly['score']:.3f}", styles["Normal"]))
        story.append(Paragraph(f"<b>Threshold:</b> {anomaly['threshold']:.3f}", styles["Normal"]))
        story.append(Paragraph(
            f"<b>What happened:</b> {summary.get('summary', 'Unusual pattern detected')}", 
            styles["Normal"]
        ))
        story.append(Spacer(1, 20))
        
        # Make a chart
        plt.figure(figsize=(7, 4))
        plt.plot(df["timestamp"], df["Close"], label="Close Price", color="#2563eb", linewidth=2)
        plt.scatter(
            df["timestamp"].iloc[-1], 
            df["Close"].iloc[-1], 
            color="#ef4444", 
            label="Anomaly", 
            s=100, 
            zorder=5
        )
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.title(f"{ticker} Price Chart")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"/tmp/{ticker}_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        story.append(Image(chart_path, width=450, height=250))
        story.append(Spacer(1, 12))
        
        # Add a table with the data
        story.append(Paragraph("<b>Data Point Details</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))
        
        table_data = [["Feature", "Value"]]
        for k, v in anomaly["latest"].items():
            if isinstance(v, (int, float, Decimal)):
                table_data.append([str(k), f"{float(v):.4f}"])
            else:
                table_data.append([str(k), str(v)])
        
        t = Table(table_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(t)
        doc.build(story)
        
        logger.info(f"Report saved: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to make report: {str(e)}")
        raise


def upload_to_s3(file_path, bucket, prefix="reports/"):
    """Upload report to S3"""
    filename = os.path.basename(file_path)
    key = f"{prefix}{filename}"
    
    try:
        s3.upload_file(file_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded to {uri}")
        return uri
    except Exception as e:
        logger.error(f"S3 upload failed: {str(e)}")
        raise


def save_to_dynamo(ticker, anomaly, report_url):
    """Store anomaly record for later analysis"""
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        item = {
            'id': f"{ticker}_{now}",
            'ticker': ticker,
            'timestamp': now,
            'anomaly_score': Decimal(str(anomaly['score'])),
            'threshold': Decimal(str(anomaly['threshold'])),
            'report_url': report_url,
            'close_price': Decimal(str(anomaly['latest'].get('Close', 0))),
            'volume': int(anomaly['latest'].get('Volume', 0))
        }
        
        table.put_item(Item=item)
        logger.info(f"Saved to DynamoDB: {item['id']}")
        
    except Exception as e:
        logger.error(f"DynamoDB write failed: {str(e)}")
        # Don't crash on this, it's not critical


# ==================== Main handler ====================

def lambda_handler(event, context=None):
    """Main entry point - gets called by Lambda or EventBridge"""
    
    ticker = event.get("ticker", TICKER) if isinstance(event, dict) else TICKER
    logger.info(f"Running detection for {ticker}")
    
    # Get the data
    df = fetch_data(ticker)
    if df.empty:
        logger.error("No data returned, bailing out")
        return {"status": "no_data", "ticker": ticker}
    
    # Add features
    df = add_features(df)
    
    # Run detection
    features = ["Open", "High", "Low", "Close", "Volume", "return", "hl_gap", "oc_gap", "volatility"]
    anomaly = check_for_anomaly(df, features)
    
    if not anomaly:
        logger.info("Everything looks normal")
        return {"status": "no_anomaly", "ticker": ticker}
    
    # Build a summary
    latest = anomaly['latest']
    price_pct = ((latest['Close'] - latest['Open']) / latest['Open']) * 100
    vol_str = f"{latest['Volume']:,.0f}"
    deviation = anomaly['score'] - anomaly['threshold']
    
    summary = {
        "summary": (
            f"Caught something unusual in {ticker}. "
            f"Price moved {price_pct:+.2f}% on volume of {vol_str}. "
            f"Anomaly score exceeded threshold by {deviation:.2f} points."
        )
    }
    
    # Make the report
    try:
        report_path = make_report(ticker, anomaly, df, summary)
        report_url = upload_to_s3(report_path, S3_BUCKET)
    except Exception as e:
        logger.error(f"Report generation bombed: {str(e)}")
        return {"status": "error", "ticker": ticker, "error": str(e)}
    
    # Store it
    save_to_dynamo(ticker, anomaly, report_url)
    
    # Send alert
    if TOPIC_ARN:
        try:
            msg = (
                f"🚨 Anomaly Alert\n\n"
                f"Ticker: {ticker}\n"
                f"Score: {anomaly['score']:.3f}\n"
                f"Threshold: {anomaly['threshold']:.3f}\n\n"
                f"{summary['summary']}\n\n"
                f"Report: {report_url}"
            )
            
            sns.publish(
                TopicArn=TOPIC_ARN,
                Message=msg,
                Subject=f"🚨 {ticker} Anomaly"
            )
            logger.info("Alert sent")
        except Exception as e:
            logger.error(f"SNS publish failed: {str(e)}")
    
    return {
        "status": "anomaly_reported",
        "ticker": ticker,
        "anomaly_score": anomaly['score'],
        "summary": summary,
        "report_url": report_url
    }


if __name__ == "__main__":
    # Quick test run
    result = lambda_handler({"ticker": "AAPL"})
    print(json.dumps(result, indent=2, default=str))