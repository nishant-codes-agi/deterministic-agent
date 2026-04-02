#!/usr/bin/env python3
"""
Stock Anomaly Detection System

This module fetches AAPL stock data and detects anomalous trading days using IQR method.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily stock data for a given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with stock data or None if fetching fails
        
    Raises:
        Exception: If data fetching fails due to network or API issues
    """
    try:
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Download historical data
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker} in the specified date range")
            
        # Ensure we have the required columns
        required_columns = ['Volume', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        print(f"Successfully fetched {len(data)} days of data")
        return data
        
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None


def detect_anomalies_iqr(df: pd.DataFrame, column: str = 'Volume') -> pd.DataFrame:
    """
    Detect anomalies in a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to analyze for anomalies
        
    Returns:
        DataFrame containing anomalies with dates, values, and percentage deviations
        
    Raises:
        ValueError: If the specified column doesn't exist or contains no valid data
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
        
    # Get the column data and remove any NaN values
    data = df[column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"No valid data found in column '{column}'")
        
    # Calculate quartiles and IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for anomaly detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"IQR Analysis for {column}:")
    print(f"Q1: {Q1:,.0f}")
    print(f"Q3: {Q3:,.0f}")
    print(f"IQR: {IQR:,.0f}")
    print(f"Lower bound: {lower_bound:,.0f}")
    print(f"Upper bound: {upper_bound:,.0f}")
    
    # Identify anomalies
    anomaly_mask = (data < lower_bound) | (data > upper_bound)
    anomalies = df[anomaly_mask].copy()
    
    if len(anomalies) == 0:
        print("No anomalies detected")
        return pd.DataFrame()
        
    # Calculate median for percentage deviation
    median_value = data.median()
    
    # Calculate percentage deviation from median
    anomalies['percentage_deviation'] = ((anomalies[column] - median_value) / median_value) * 100
    
    # Add absolute deviation for sorting
    anomalies['abs_deviation'] = abs(anomalies['percentage_deviation'])
    
    print(f"Found {len(anomalies)} anomalies")
    
    return anomalies


def print_anomaly_summary(anomalies: pd.DataFrame, column: str = 'Volume', top_n: int = 5) -> None:
    """
    Print a formatted summary of the top anomalies.
    
    Args:
        anomalies: DataFrame containing anomaly data
        column: Column name that was analyzed
        top_n: Number of top anomalies to display
    """
    if anomalies.empty:
        print("No anomalies to display")
        return
        
    # Sort by absolute deviation and get top N
    top_anomalies = anomalies.nlargest(top_n, 'abs_deviation')
    
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(anomalies))} ANOMALOUS TRADING DAYS FOR AAPL")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Date':<12} {column:<15} {'Deviation':<12} {'Type':<10}")
    print(f"{'-'*80}")
    
    for i, (date, row) in enumerate(top_anomalies.iterrows(), 1):
        volume = row[column]
        deviation = row['percentage_deviation']
        anomaly_type = "High" if deviation > 0 else "Low"
        
        # Format date
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        
        print(f"{i:<4} {date_str:<12} {volume:>13,.0f} {deviation:>+10.1f}% {anomaly_type:<10}")
    
    print(f"{'-'*80}")
    print(f"Total anomalies detected: {len(anomalies)}")
    print(f"Analysis period: {anomalies.index.min().strftime('%Y-%m-%d')} to {anomalies.index.max().strftime('%Y-%m-%d')}")


def main() -> None:
    """
    Main execution function that orchestrates the anomaly detection process.
    """
    # Define constants
    TICKER = 'AAPL'
    
    # Calculate date range (2 years from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # Approximately 2 years
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Stock Anomaly Detection System")
    print(f"Analyzing {TICKER} from {start_date_str} to {end_date_str}")
    print("="*60)
    
    try:
        # Step 1: Fetch stock data
        stock_data = fetch_stock_data(TICKER, start_date_str, end_date_str)
        
        if stock_data is None:
            print("Failed to fetch stock data. Exiting.")
            return
            
        # Step 2: Detect anomalies using IQR method
        print("\nDetecting anomalies using IQR method...")
        anomalies = detect_anomalies_iqr(stock_data, column='Volume')
        
        # Step 3: Print summary of top anomalies
        print_anomaly_summary(anomalies, column='Volume', top_n=5)
        
        # Additional statistics
        if not anomalies.empty:
            print(f"\nAdditional Statistics:")
            print(f"Average volume: {stock_data['Volume'].mean():,.0f}")
            print(f"Median volume: {stock_data['Volume'].median():,.0f}")
            print(f"Standard deviation: {stock_data['Volume'].std():,.0f}")
            print(f"Anomaly rate: {len(anomalies)/len(stock_data)*100:.2f}%")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
