#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

This script fetches stock data and detects anomalies using the IQR method.
It generates a visualization and saves results to JSON.
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using: pip install -r requirements.txt")
    sys.exit(1)

# Constants
TICKER = 'AAPL'
START_DATE = '2022-01-01'
END_DATE = '2023-01-01'
ANOMALY_COLUMN = 'Close'
IQR_MULTIPLIER = 1.5
CHART_OUTPUT = 'anomalies.png'
JSON_OUTPUT = 'anomaly_report.json'

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with stock data or None if error
    """
    try:
        print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"Error: No data found for ticker {ticker}")
            return None
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Flatten column names if they are MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == ticker else f"{col[0]}_{col[1]}" for col in data.columns]
        
        print(f"Successfully fetched {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None

def detect_anomalies_iqr(data: pd.DataFrame, column: str, multiplier: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Detect anomalies using the IQR method.
    
    Args:
        data: DataFrame with stock data
        column: Column name to analyze for anomalies
        multiplier: IQR multiplier for threshold calculation
        
    Returns:
        Tuple of (anomalies_df, stats_dict)
    """
    try:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        values = data[column].dropna()
        
        if len(values) == 0:
            print("Warning: No valid data points found")
            return pd.DataFrame(), {}
            
        # Calculate IQR statistics
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        print(f"IQR Analysis for {column}:")
        print(f"  Q1: ${q1:.2f}")
        print(f"  Q3: ${q3:.2f}")
        print(f"  IQR: ${iqr:.2f}")
        print(f"  Lower bound: ${lower_bound:.2f}")
        print(f"  Upper bound: ${upper_bound:.2f}")
        
        # Create boolean masks for anomalies
        lower_mask = values < lower_bound
        upper_mask = values > upper_bound
        anomaly_mask = lower_mask | upper_mask
        
        # Get anomalies using boolean indexing
        anomaly_indices = values[anomaly_mask].index
        anomalies = data.loc[anomaly_indices].copy()
        
        if len(anomalies) > 0:
            anomalies['anomaly_type'] = 'normal'
            anomalies.loc[anomalies[column] < lower_bound, 'anomaly_type'] = 'low'
            anomalies.loc[anomalies[column] > upper_bound, 'anomaly_type'] = 'high'
        
        stats = {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'total_points': len(values),
            'anomaly_count': len(anomalies)
        }
        
        return anomalies, stats
        
    except Exception as e:
        print(f"Error in anomaly detection: {str(e)}")
        return pd.DataFrame(), {}

def calculate_daily_returns(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    Calculate daily returns for the stock data.
    
    Args:
        data: DataFrame with stock data
        price_column: Column name for price data
        
    Returns:
        DataFrame with daily returns added
    """
    try:
        data_copy = data.copy()
        data_copy['daily_return'] = data_copy[price_column].pct_change() * 100
        return data_copy
    except Exception as e:
        print(f"Error calculating daily returns: {str(e)}")
        return data

def create_visualization(data: pd.DataFrame, anomalies: pd.DataFrame, column: str, stats: Dict[str, float]) -> bool:
    """
    Create and save a visualization of the data with anomalies highlighted.
    
    Args:
        data: Full dataset
        anomalies: Detected anomalies
        column: Column being analyzed
        stats: Statistics from anomaly detection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Convert Date column to datetime if it's not already
        if 'Date' in data.columns:
            dates = pd.to_datetime(data['Date'])
        else:
            dates = data.index
            
        # Plot main data
        plt.plot(dates, data[column], label=f'{TICKER} {column} Price', color='blue', alpha=0.7)
        
        # Plot anomalies if any exist
        if len(anomalies) > 0:
            if 'Date' in anomalies.columns:
                anomaly_dates = pd.to_datetime(anomalies['Date'])
            else:
                anomaly_dates = anomalies.index
                
            plt.scatter(anomaly_dates, anomalies[column], 
                       color='red', s=50, label=f'Anomalies ({len(anomalies)})', zorder=5)
        
        # Add threshold lines
        if stats:
            plt.axhline(y=stats['lower_bound'], color='orange', linestyle='--', alpha=0.7, label='Lower Threshold')
            plt.axhline(y=stats['upper_bound'], color='orange', linestyle='--', alpha=0.7, label='Upper Threshold')
        
        plt.title(f'{TICKER} Stock Price Anomaly Detection\nUsing IQR Method (Multiplier: {IQR_MULTIPLIER})', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{column} Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(CHART_OUTPUT, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {CHART_OUTPUT}")
        return True
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False

def save_json_report(data: pd.DataFrame, anomalies: pd.DataFrame, stats: Dict[str, float]) -> bool:
    """
    Save anomaly detection results to JSON file.
    
    Args:
        data: Full dataset
        anomalies: Detected anomalies
        stats: Statistics from anomaly detection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Prepare anomaly records
        anomaly_records = []
        
        if len(anomalies) > 0:
            for idx, row in anomalies.iterrows():
                record = {
                    'date': row['Date'].strftime('%Y-%m-%d') if 'Date' in row else str(idx),
                    'close_price': float(row[ANOMALY_COLUMN]),
                    'anomaly_type': row.get('anomaly_type', 'unknown')
                }
                
                # Add daily return if available
                if 'daily_return' in row and pd.notna(row['daily_return']):
                    record['daily_return'] = float(row['daily_return'])
                else:
                    record['daily_return'] = None
                    
                anomaly_records.append(record)
        
        # Create report
        report = {
            'metadata': {
                'ticker': TICKER,
                'analysis_period': {
                    'start_date': START_DATE,
                    'end_date': END_DATE
                },
                'method': 'IQR',
                'iqr_multiplier': IQR_MULTIPLIER,
                'analyzed_column': ANOMALY_COLUMN,
                'generated_at': datetime.now().isoformat()
            },
            'statistics': stats,
            'anomalies': anomaly_records,
            'summary': {
                'total_data_points': len(data),
                'anomalies_detected': len(anomalies),
                'anomaly_percentage': round((len(anomalies) / len(data)) * 100, 2) if len(data) > 0 else 0
            }
        }
        
        with open(JSON_OUTPUT, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"JSON report saved to {JSON_OUTPUT}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON report: {str(e)}")
        return False

def print_anomaly_summary(anomalies: pd.DataFrame, stats: Dict[str, float]) -> None:
    """
    Print a summary of detected anomalies to console.
    
    Args:
        anomalies: Detected anomalies DataFrame
        stats: Statistics from anomaly detection
    """
    try:
        print("\n" + "="*50)
        print("ANOMALY DETECTION SUMMARY")
        print("="*50)
        
        if len(anomalies) == 0:
            print("No anomalies detected in the data.")
        else:
            print(f"Total anomalies detected: {len(anomalies)}")
            print(f"Anomaly rate: {(len(anomalies) / stats.get('total_points', 1)) * 100:.2f}%")
            print("\nDetailed anomalies:")
            print("-" * 30)
            
            for idx, row in anomalies.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d') if 'Date' in row else str(idx)
                price = row[ANOMALY_COLUMN]
                anomaly_type = row.get('anomaly_type', 'unknown')
                
                print(f"Date: {date_str}")
                print(f"  {ANOMALY_COLUMN}: ${price:.2f}")
                print(f"  Type: {anomaly_type.upper()}")
                
                if 'daily_return' in row and pd.notna(row['daily_return']):
                    print(f"  Daily Return: {row['daily_return']:.2f}%")
                print()
                
    except Exception as e:
        print(f"Error printing summary: {str(e)}")

def main() -> None:
    """
    Main function to orchestrate the anomaly detection process.
    """
    print(f"Stock Anomaly Detection Tool")
    print(f"Analyzing {TICKER} stock data...")
    print()
    
    # Fetch stock data
    data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    if data is None:
        print("Failed to fetch stock data. Exiting.")
        sys.exit(1)
    
    # Calculate daily returns
    data = calculate_daily_returns(data, ANOMALY_COLUMN)
    
    # Detect anomalies
    anomalies, stats = detect_anomalies_iqr(data, ANOMALY_COLUMN, IQR_MULTIPLIER)
    
    # Print summary to console
    print_anomaly_summary(anomalies, stats)
    
    # Create visualization
    viz_success = create_visualization(data, anomalies, ANOMALY_COLUMN, stats)
    
    # Save JSON report
    json_success = save_json_report(data, anomalies, stats)
    
    # Final status
    print("\n" + "="*50)
    print("PROCESS COMPLETED")
    print("="*50)
    print(f"Data fetched: ✓")
    print(f"Anomalies detected: ✓ ({len(anomalies)} found)")
    print(f"Visualization: {'✓' if viz_success else '✗'}")
    print(f"JSON report: {'✓' if json_success else '✗'}")
    
    if not viz_success or not json_success:
        print("\nSome outputs failed to generate. Check error messages above.")
        sys.exit(1)
    
    print(f"\nFiles generated:")
    print(f"  - {CHART_OUTPUT}")
    print(f"  - {JSON_OUTPUT}")

if __name__ == "__main__":
    main()
