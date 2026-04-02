#!/usr/bin/env python3
"""
AAPL Stock Anomaly Detection

Fetches 2 weeks of daily AAPL stock data, detects anomalous trading days using IQR,
and prints a summary of the top 5 anomalies with their dates and percentage deviations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker='AAPL', days=14):
    """
    Fetch historical stock data for the specified ticker and number of days.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of trading days to fetch
    
    Returns:
        pd.DataFrame: Stock data with Date index and Close prices
    """
    try:
        # Calculate date range - fetch extra days to account for weekends/holidays
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)  # Buffer for weekends
        
        print(f"Fetching {ticker} stock data from {start_date.date()} to {end_date.date()}...")
        
        # Download stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Keep only the most recent trading days
        data = data.tail(days)
        
        print(f"Successfully fetched {len(data)} trading days of data")
        return data[['Close']]
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        sys.exit(1)

def calculate_percentage_changes(data):
    """
    Calculate daily percentage changes in stock price.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
    
    Returns:
        pd.Series: Daily percentage changes
    """
    try:
        pct_changes = data['Close'].pct_change().dropna() * 100
        print(f"Calculated percentage changes for {len(pct_changes)} trading days")
        return pct_changes
    except Exception as e:
        print(f"Error calculating percentage changes: {e}")
        sys.exit(1)

def detect_anomalies_iqr(pct_changes, multiplier=1.5):
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Args:
        pct_changes (pd.Series): Daily percentage changes
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        tuple: (anomalies_series, q1, q3, iqr, lower_bound, upper_bound)
    """
    try:
        # Calculate quartiles and IQR
        q1 = pct_changes.quantile(0.25)
        q3 = pct_changes.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Identify anomalies
        anomalies = pct_changes[(pct_changes < lower_bound) | (pct_changes > upper_bound)]
        
        print(f"\nIQR Analysis:")
        print(f"Q1: {q1:.2f}%")
        print(f"Q3: {q3:.2f}%")
        print(f"IQR: {iqr:.2f}%")
        print(f"Lower bound: {lower_bound:.2f}%")
        print(f"Upper bound: {upper_bound:.2f}%")
        print(f"Found {len(anomalies)} anomalous trading days")
        
        return anomalies, q1, q3, iqr, lower_bound, upper_bound
        
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        sys.exit(1)

def get_top_anomalies(anomalies, top_n=5):
    """
    Get the top N anomalies sorted by absolute percentage deviation.
    
    Args:
        anomalies (pd.Series): Anomalous percentage changes
        top_n (int): Number of top anomalies to return
    
    Returns:
        pd.Series: Top N anomalies sorted by absolute deviation
    """
    try:
        if len(anomalies) == 0:
            print("No anomalies found to rank")
            return pd.Series(dtype=float)
        
        # Sort by absolute value of percentage change
        top_anomalies = anomalies.reindex(anomalies.abs().sort_values(ascending=False).index)
        
        # Return top N
        return top_anomalies.head(top_n)
        
    except Exception as e:
        print(f"Error ranking anomalies: {e}")
        return pd.Series(dtype=float)

def print_anomaly_summary(top_anomalies, ticker='AAPL'):
    """
    Print a formatted summary of the top anomalies.
    
    Args:
        top_anomalies (pd.Series): Top anomalies to summarize
        ticker (str): Stock ticker symbol
    """
    try:
        print(f"\n{'='*60}")
        print(f"TOP {len(top_anomalies)} ANOMALOUS TRADING DAYS FOR {ticker}")
        print(f"{'='*60}")
        
        if len(top_anomalies) == 0:
            print("No anomalies detected in the analyzed period.")
            return
        
        print(f"{'Rank':<4} {'Date':<12} {'Change (%)':<12} {'Type':<10}")
        print(f"{'-'*4} {'-'*12} {'-'*12} {'-'*10}")
        
        for rank, (date, pct_change) in enumerate(top_anomalies.items(), 1):
            anomaly_type = "Spike" if pct_change > 0 else "Drop"
            date_str = date.strftime('%Y-%m-%d')
            print(f"{rank:<4} {date_str:<12} {pct_change:>+8.2f}%    {anomaly_type:<10}")
        
        print(f"\nSummary Statistics:")
        print(f"Largest gain: +{top_anomalies.max():.2f}%")
        print(f"Largest drop: {top_anomalies.min():.2f}%")
        print(f"Average absolute deviation: {top_anomalies.abs().mean():.2f}%")
        
    except Exception as e:
        print(f"Error printing summary: {e}")

def main():
    """
    Main function to orchestrate the anomaly detection process.
    """
    try:
        print("AAPL Stock Anomaly Detection System")
        print("===================================\n")
        
        # Configuration
        TICKER = 'AAPL'
        DAYS = 14
        IQR_MULTIPLIER = 1.5
        TOP_N = 5
        
        # Step 1: Fetch stock data
        stock_data = fetch_stock_data(TICKER, DAYS)
        
        # Step 2: Calculate percentage changes
        pct_changes = calculate_percentage_changes(stock_data)
        
        # Step 3: Detect anomalies using IQR
        anomalies, q1, q3, iqr, lower_bound, upper_bound = detect_anomalies_iqr(
            pct_changes, IQR_MULTIPLIER
        )
        
        # Step 4: Get top anomalies
        top_anomalies = get_top_anomalies(anomalies, TOP_N)
        
        # Step 5: Print summary
        print_anomaly_summary(top_anomalies, TICKER)
        
        print(f"\nAnalysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
