#!/usr/bin/env python3
"""
TSLA Stock Anomaly Detection

This script fetches TSLA stock data for the last 14 days,
detects anomalous trading days based on Z-score analysis,
and generates comprehensive reports.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime, timedelta
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker, period='14d'):
    """
    Fetch stock data from yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for data retrieval
    
    Returns:
        pd.DataFrame or None: Stock data or None if error occurs
    """
    try:
        print(f"Fetching {ticker} stock data for the last {period}...")
        
        # Download data with error handling
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"Error: No data found for ticker {ticker}")
            return None
            
        print(f"Successfully fetched {len(df)} days of data")
        return df
        
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None


def calculate_anomalies(df, z_threshold=2.0):
    """
    Calculate daily returns and identify anomalous trading days.
    
    Args:
        df (pd.DataFrame): Stock data with 'Close' column
        z_threshold (float): Z-score threshold for anomaly detection
    
    Returns:
        tuple: (anomalies_df, all_data_df) containing anomalous days and all data with calculations
    """
    try:
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Calculate daily returns using adjusted close price
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Drop the first row with NaN return
        data = data.dropna()
        
        if len(data) < 2:
            print("Warning: Insufficient data for anomaly detection")
            return pd.DataFrame(), data
        
        # Calculate statistics for the entire period
        mean_return = data['Daily_Return'].mean()
        std_return = data['Daily_Return'].std()
        
        if std_return == 0:
            print("Warning: Standard deviation is zero, no anomalies can be detected")
            data['Z_Score'] = 0
            return pd.DataFrame(), data
        
        # Calculate Z-scores
        data['Z_Score'] = (data['Daily_Return'] - mean_return) / std_return
        
        # Identify anomalies
        anomaly_mask = (np.abs(data['Z_Score']) > z_threshold)
        anomalies = data[anomaly_mask].copy()
        
        # Add anomaly flag to all data
        data['Is_Anomaly'] = anomaly_mask
        
        print(f"Found {len(anomalies)} anomalous trading days out of {len(data)} total days")
        
        return anomalies, data
        
    except Exception as e:
        print(f"Error calculating anomalies: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def generate_markdown_report(anomalies_df, all_data_df, ticker):
    """
    Generate a markdown summary report.
    
    Args:
        anomalies_df (pd.DataFrame): DataFrame containing anomalous days
        all_data_df (pd.DataFrame): DataFrame containing all trading data
        ticker (str): Stock ticker symbol
    
    Returns:
        str: Markdown formatted report
    """
    report_lines = []
    report_lines.append(f"# {ticker} Stock Anomaly Detection Report")
    report_lines.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Analysis Period:** Last 14 days")
    report_lines.append(f"\n**Total Trading Days Analyzed:** {len(all_data_df)}")
    
    if len(anomalies_df) == 0:
        report_lines.append("\n## Summary")
        report_lines.append("\n✅ **No anomalous trading days detected** (Z-score threshold: ±2.0)")
        report_lines.append("\nAll daily returns fall within normal statistical ranges.")
    else:
        report_lines.append("\n## Summary")
        report_lines.append(f"\n⚠️ **{len(anomalies_df)} anomalous trading day(s) detected** (Z-score threshold: ±2.0)")
        
        report_lines.append("\n## Anomalous Trading Days")
        report_lines.append("\n| Date | Close Price | Daily Return (%) | Z-Score | Severity |")
        report_lines.append("|------|-------------|------------------|---------|----------|")
        
        for idx, row in anomalies_df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            close_price = f"${row['Close']:.2f}"
            daily_return = f"{row['Daily_Return']*100:.2f}%"
            z_score = f"{row['Z_Score']:.2f}"
            
            # Determine severity
            abs_z = abs(row['Z_Score'])
            if abs_z >= 3.0:
                severity = "🔴 Extreme"
            elif abs_z >= 2.5:
                severity = "🟠 High"
            else:
                severity = "🟡 Moderate"
            
            report_lines.append(f"| {date_str} | {close_price} | {daily_return} | {z_score} | {severity} |")
    
    # Add statistics section
    if len(all_data_df) > 0:
        report_lines.append("\n## Statistical Summary")
        mean_return = all_data_df['Daily_Return'].mean() * 100
        std_return = all_data_df['Daily_Return'].std() * 100
        min_return = all_data_df['Daily_Return'].min() * 100
        max_return = all_data_df['Daily_Return'].max() * 100
        
        report_lines.append(f"\n- **Mean Daily Return:** {mean_return:.2f}%")
        report_lines.append(f"- **Standard Deviation:** {std_return:.2f}%")
        report_lines.append(f"- **Minimum Daily Return:** {min_return:.2f}%")
        report_lines.append(f"- **Maximum Daily Return:** {max_return:.2f}%")
    
    report_lines.append("\n## Methodology")
    report_lines.append("\nAnomalies are detected using Z-score analysis:")
    report_lines.append("\n1. Calculate daily returns as percentage change in closing price")
    report_lines.append("2. Compute mean and standard deviation of daily returns")
    report_lines.append("3. Calculate Z-score: (return - mean) / standard_deviation")
    report_lines.append("4. Flag days where |Z-score| > 2.0 as anomalous")
    
    report_lines.append("\n---")
    report_lines.append("\n*Report generated by TSLA Stock Anomaly Detection System*")
    
    return "\n".join(report_lines)


def create_visualization(all_data_df, anomalies_df, ticker):
    """
    Create and save a matplotlib chart highlighting anomalous days.
    
    Args:
        all_data_df (pd.DataFrame): All trading data
        anomalies_df (pd.DataFrame): Anomalous trading days
        ticker (str): Stock ticker symbol
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Stock price with anomalous days highlighted
        ax1.plot(all_data_df.index, all_data_df['Close'], 'b-', linewidth=2, label='Close Price')
        
        if len(anomalies_df) > 0:
            ax1.scatter(anomalies_df.index, anomalies_df['Close'], 
                       color='red', s=100, zorder=5, label='Anomalous Days')
        
        ax1.set_title(f'{ticker} Stock Price - Last 14 Days', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        # Plot 2: Daily returns with Z-score threshold lines
        returns_pct = all_data_df['Daily_Return'] * 100
        ax2.bar(all_data_df.index, returns_pct, alpha=0.7, color='lightblue', label='Daily Returns')
        
        if len(anomalies_df) > 0:
            anomaly_returns = anomalies_df['Daily_Return'] * 100
            ax2.bar(anomalies_df.index, anomaly_returns, color='red', alpha=0.8, label='Anomalous Returns')
        
        # Add Z-score threshold lines (approximate)
        if len(all_data_df) > 0:
            mean_return = all_data_df['Daily_Return'].mean() * 100
            std_return = all_data_df['Daily_Return'].std() * 100
            
            upper_threshold = mean_return + 2 * std_return
            lower_threshold = mean_return - 2 * std_return
            
            ax2.axhline(y=upper_threshold, color='orange', linestyle='--', alpha=0.7, label='Z-score = +2')
            ax2.axhline(y=lower_threshold, color='orange', linestyle='--', alpha=0.7, label='Z-score = -2')
            ax2.axhline(y=mean_return, color='green', linestyle='-', alpha=0.5, label='Mean Return')
        
        ax2.set_title('Daily Returns with Anomaly Thresholds', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Daily Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        # Save the chart
        plt.savefig('anomalies.png', dpi=300, bbox_inches='tight')
        print("Chart saved as 'anomalies.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")


def save_json_report(anomalies_df, all_data_df, ticker):
    """
    Save anomaly report as JSON file.
    
    Args:
        anomalies_df (pd.DataFrame): Anomalous trading days
        all_data_df (pd.DataFrame): All trading data
        ticker (str): Stock ticker symbol
    """
    try:
        report_data = {
            "ticker": ticker,
            "analysis_date": datetime.now().isoformat(),
            "period": "14d",
            "total_trading_days": len(all_data_df),
            "anomalous_days_count": len(anomalies_df),
            "z_score_threshold": 2.0,
            "statistics": {},
            "anomalous_days": []
        }
        
        # Add statistics if data is available
        if len(all_data_df) > 0:
            report_data["statistics"] = {
                "mean_daily_return": float(all_data_df['Daily_Return'].mean()),
                "std_daily_return": float(all_data_df['Daily_Return'].std()),
                "min_daily_return": float(all_data_df['Daily_Return'].min()),
                "max_daily_return": float(all_data_df['Daily_Return'].max())
            }
        
        # Add anomalous days data
        for idx, row in anomalies_df.iterrows():
            anomaly_data = {
                "date": idx.strftime('%Y-%m-%d'),
                "close_price": float(row['Close']),
                "daily_return": float(row['Daily_Return']),
                "z_score": float(row['Z_Score']),
                "severity": "extreme" if abs(row['Z_Score']) >= 3.0 else "high" if abs(row['Z_Score']) >= 2.5 else "moderate"
            }
            report_data["anomalous_days"].append(anomaly_data)
        
        # Save to JSON file
        with open('anomaly_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print("JSON report saved as 'anomaly_report.json'")
        
    except Exception as e:
        print(f"Error saving JSON report: {str(e)}")


def main():
    """
    Main execution function.
    """
    ticker = 'TSLA'
    period = '14d'
    
    print("=" * 50)
    print("TSLA Stock Anomaly Detection System")
    print("=" * 50)
    
    # Step 1: Fetch stock data
    stock_data = fetch_stock_data(ticker, period)
    
    if stock_data is None:
        print("Failed to fetch stock data. Exiting.")
        return
    
    # Step 2: Calculate anomalies
    anomalies_df, all_data_df = calculate_anomalies(stock_data)
    
    if all_data_df.empty:
        print("Failed to process stock data. Exiting.")
        return
    
    # Step 3: Generate markdown report
    markdown_report = generate_markdown_report(anomalies_df, all_data_df, ticker)
    
    # Step 4: Create visualization
    create_visualization(all_data_df, anomalies_df, ticker)
    
    # Step 5: Save JSON report
    save_json_report(anomalies_df, all_data_df, ticker)
    
    # Step 6: Display markdown report
    print("\n" + "=" * 50)
    print("MARKDOWN REPORT")
    print("=" * 50)
    print(markdown_report)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files generated:")
    print("- anomalies.png (visualization chart)")
    print("- anomaly_report.json (detailed JSON report)")


if __name__ == "__main__":
    main()