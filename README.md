# Project: Financial Anomaly Detection & Alerting System with AWS + LLM Summarization

A real-time monitoring system that detects unusual patterns in financial market data using machine learning. Built for AWS Lambda, this system continuously analyzes stock prices and trading volumes, automatically alerting you when something weird is happening.

### What It Does
Think of this as your automated market watchdog. Every time it runs:

Pulls fresh data from Yahoo Finance (stock prices, volumes, etc.),
Calculates technical features like volatility, price gaps, and returns,
Runs ML models (Isolation Forest + Local Outlier Factor) to spot anomalies,
Generates a PDF report with charts and analysis,
Stores everything in DynamoDB for historical tracking,
Sends alerts via SNS when something unusual is detected,

### Why This Exists
Market anomalies can signal important events - sudden price spikes, unusual trading volumes, or weird patterns that might indicate news events, algorithm glitches, or manipulation. This system catches those moments automatically so you don't have to stare at charts all day.
