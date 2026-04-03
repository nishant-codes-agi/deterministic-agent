#!/usr/bin/env python3
"""
Stock Anomaly Detection Script

This script fetches 1 year of daily OHLCV data for AAPL using yfinance,
detects anomalous trading days using Z-score analysis on log returns,
and produces a visualization and JSON report of the anomalies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from typing import Optional, Dict, List, Any

# Constants
TICKER = 'AAPL'
Z_SCORE_THRESHOLD = 3.0
CHART_FILENAME = 'anomalies.png'
REPORT_FILENAME = 'anomaly_report.json'


def fetch_stock_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from yfinance for the specified date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with OHLCV data or None if error occurs
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None
            
        print(f"Successfully fetched {len(data)} days of data")
        return data
        
    except KeyError as e:
        print(f"Invalid ticker symbol {ticker}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalous trading days using Z-score analysis on log returns.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional columns for log returns, Z-scores, and anomaly flags
    """
    print("Calculating log returns and detecting anomalies...")
    
    # Create a copy to avoid modifying original data
    df_analysis = df.copy()
    
    # Calculate log returns
    df_analysis['Log_Returns'] = np.log(df_analysis['Close'] / df_analysis['Close'].shift(1))
    
    # Drop NaN values (first row will be NaN due to shift)
    df_analysis = df_analysis.dropna()
    
    # Calculate Z-scores
    mean_return = df_analysis['Log_Returns'].mean()
    std_return = df_analysis['Log_Returns'].std()
    
    df_analysis['Z_Score'] = (df_analysis['Log_Returns'] - mean_return) / std_return
    
    # Identify anomalies
    df_analysis['Is_Anomaly'] = np.abs(df_analysis['Z_Score']) > Z_SCORE_THRESHOLD
    
    anomaly_count = df_analysis['Is_Anomaly'].sum()
    print(f"Detected {anomaly_count} anomalous trading days")
    
    return df_analysis


def plot_anomalies(df_anomalies: pd.DataFrame, filename: str) -> None:
    """
    Create and save a matplotlib chart highlighting anomalous days.
    
    Args:
        df_anomalies: DataFrame with anomaly detection results
        filename: Output filename for the chart
    """
    print(f"Creating anomaly visualization chart: {filename}")
    
    plt.figure(figsize=(12, 8))
    
    # Plot the close price series
    plt.plot(df_anomalies.index, df_anomalies['Close'], 
             label='Close Price', color='blue', linewidth=1)
    
    # Highlight anomalous days
    anomalous_data = df_anomalies[df_anomalies['Is_Anomaly']]
    if not anomalous_data.empty:
        plt.scatter(anomalous_data.index, anomalous_data['Close'], 
                   color='red', s=50, marker='o', 
                   label=f'Anomalous Days ({len(anomalous_data)})', zorder=5)
    
    plt.title(f'{TICKER} Stock Price with Anomalous Trading Days\n'
              f'(Z-score threshold: {Z_SCORE_THRESHOLD})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {filename}")


def generate_report(df_anomalies: pd.DataFrame, filename: str) -> None:
    """
    Generate a JSON report of anomalous trading days.
    
    Args:
        df_anomalies: DataFrame with anomaly detection results
        filename: Output filename for the JSON report
    """
    print(f"Generating anomaly report: {filename}")
    
    # Filter for anomalous days only
    anomalous_days = df_anomalies[df_anomalies['Is_Anomaly']]
    
    # Create report data
    report_data = {
        'summary': {
            'ticker': TICKER,
            'analysis_period': {
                'start_date': df_anomalies.index.min().strftime('%Y-%m-%d'),
                'end_date': df_anomalies.index.max().strftime('%Y-%m-%d')
            },
            'total_trading_days': len(df_anomalies),
            'anomalous_days_count': len(anomalous_days),
            'anomaly_percentage': round((len(anomalous_days) / len(df_anomalies)) * 100, 2),
            'z_score_threshold': Z_SCORE_THRESHOLD
        },
        'anomalies': []
    }
    
    # Add individual anomaly records
    for date, row in anomalous_days.iterrows():
        anomaly_record = {
            'date': date.strftime('%Y-%m-%d'),
            'return_value': round(row['Log_Returns'], 6),
            'z_score': round(row['Z_Score'], 3),
            'close_price': round(row['Close'], 2),
            'volume': int(row['Volume'])
        }
        report_data['anomalies'].append(anomaly_record)
    
    # Sort anomalies by absolute Z-score (most extreme first)
    report_data['anomalies'].sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    # Write to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Error saving report: {e}")


def main() -> None:
    """
    Main function to orchestrate the anomaly detection process.
    """
    print("=== Stock Anomaly Detection Analysis ===")
    print(f"Analyzing {TICKER} for anomalous trading days\n")
    
    # Calculate date range (1 year ending today)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    # Fetch stock data
    stock_data = fetch_stock_data(TICKER, start_date, end_date)
    
    if stock_data is None:
        print("Failed to fetch stock data. Exiting.")
        return
    
    # Detect anomalies
    try:
        df_with_anomalies = detect_anomalies(stock_data)
        
        # Generate visualization
        plot_anomalies(df_with_anomalies, CHART_FILENAME)
        
        # Generate JSON report
        generate_report(df_with_anomalies, REPORT_FILENAME)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to:")
        print(f"  - Chart: {CHART_FILENAME}")
        print(f"  - Report: {REPORT_FILENAME}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return


if __name__ == "__main__":
    main()
