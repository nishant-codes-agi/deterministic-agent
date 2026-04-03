#!/usr/bin/env python3
"""
META Stock Anomaly Detection

This script fetches META stock data for the last 14 days,
detects anomalous trading days using Z-score analysis,
and generates comprehensive reports.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any


def fetch_stock_data(ticker: str, days: int = 14) -> pd.DataFrame:
    """
    Fetch stock data for the specified ticker and number of days.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to fetch (default: 14)
        
    Returns:
        DataFrame with stock data
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 5)  # Extra days to ensure we get 14 trading days
        
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Get the last 14 trading days
        data = data.tail(14)
        
        return data
        
    except Exception as e:
        raise Exception(f"Failed to fetch stock data: {str(e)}")


def calculate_anomalies(data: pd.DataFrame, threshold: float = 2.0) -> Dict[str, Any]:
    """
    Calculate daily returns and detect anomalies using Z-score.
    
    Args:
        data: Stock data DataFrame
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Remove NaN values
    returns = data['Daily_Return'].dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data available")
    
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        raise ValueError("Standard deviation is zero - cannot calculate Z-scores")
    
    # Calculate Z-scores
    data['Z_Score'] = (data['Daily_Return'] - mean_return) / std_return
    
    # Identify anomalies
    data['Is_Anomaly'] = (abs(data['Z_Score']) > threshold)
    
    # Get anomalous days
    anomalous_days = data[data['Is_Anomaly'] == True].copy()
    
    return {
        'data': data,
        'anomalous_days': anomalous_days,
        'mean_return': mean_return,
        'std_return': std_return,
        'threshold': threshold,
        'total_days': len(data),
        'anomaly_count': len(anomalous_days)
    }


def create_visualization(analysis_results: Dict[str, Any], ticker: str) -> None:
    """
    Create and save a matplotlib chart highlighting anomalous days.
    
    Args:
        analysis_results: Results from anomaly analysis
        ticker: Stock ticker symbol
    """
    data = analysis_results['data']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Stock price with anomalous days highlighted
    ax1.plot(data.index, data['Close'], 'b-', linewidth=2, label='Close Price')
    
    # Highlight anomalous days
    anomalous_data = data[data['Is_Anomaly'] == True]
    if not anomalous_data.empty:
        ax1.scatter(anomalous_data.index, anomalous_data['Close'], 
                   color='red', s=100, zorder=5, label='Anomalous Days')
    
    ax1.set_title(f'{ticker} Stock Price - Last 14 Trading Days')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily returns with Z-scores
    ax2.bar(data.index, data['Daily_Return'] * 100, alpha=0.7, label='Daily Returns (%)')
    
    # Highlight anomalous returns
    if not anomalous_data.empty:
        ax2.bar(anomalous_data.index, anomalous_data['Daily_Return'] * 100, 
               color='red', alpha=0.8, label='Anomalous Returns')
    
    # Add threshold lines
    threshold = analysis_results['threshold']
    mean_return = analysis_results['mean_return']
    std_return = analysis_results['std_return']
    
    upper_threshold = (mean_return + threshold * std_return) * 100
    lower_threshold = (mean_return - threshold * std_return) * 100
    
    ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score = +{threshold}')
    ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score = -{threshold}')
    
    ax2.set_title('Daily Returns with Anomaly Thresholds')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('anomalies.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_json_report(analysis_results: Dict[str, Any], ticker: str) -> None:
    """
    Create and save a JSON report with anomaly details.
    
    Args:
        analysis_results: Results from anomaly analysis
        ticker: Stock ticker symbol
    """
    anomalous_days = analysis_results['anomalous_days']
    
    # Prepare anomaly data for JSON serialization
    anomalies = []
    for date, row in anomalous_days.iterrows():
        anomalies.append({
            'date': date.strftime('%Y-%m-%d'),
            'close_price': float(row['Close']),
            'daily_return': float(row['Daily_Return']),
            'z_score': float(row['Z_Score']),
            'return_percentage': float(row['Daily_Return'] * 100)
        })
    
    report = {
        'ticker': ticker,
        'analysis_period': {
            'start_date': analysis_results['data'].index[0].strftime('%Y-%m-%d'),
            'end_date': analysis_results['data'].index[-1].strftime('%Y-%m-%d'),
            'total_trading_days': int(analysis_results['total_days'])
        },
        'statistics': {
            'mean_daily_return': float(analysis_results['mean_return']),
            'std_daily_return': float(analysis_results['std_return']),
            'z_score_threshold': float(analysis_results['threshold'])
        },
        'anomaly_summary': {
            'total_anomalies': int(analysis_results['anomaly_count']),
            'anomaly_rate': float(analysis_results['anomaly_count'] / analysis_results['total_days'])
        },
        'anomalous_days': anomalies,
        'generated_at': datetime.now().isoformat()
    }
    
    with open('anomaly_report.json', 'w') as f:
        json.dump(report, f, indent=2)


def create_html_report(analysis_results: Dict[str, Any], ticker: str) -> None:
    """
    Create and save an HTML report.
    
    Args:
        analysis_results: Results from anomaly analysis
        ticker: Stock ticker symbol
    """
    data = analysis_results['data']
    anomalous_days = analysis_results['anomalous_days']
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{ticker} Stock Anomaly Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .anomaly {{ background-color: #ffebee; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .stats {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{ticker} Stock Anomaly Detection Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Analysis Summary</h2>
        <div class="stats">
            <p><strong>Analysis Period:</strong> {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}</p>
            <p><strong>Total Trading Days:</strong> {analysis_results['total_days']}</p>
            <p><strong>Anomalous Days Detected:</strong> {analysis_results['anomaly_count']}</p>
            <p><strong>Anomaly Rate:</strong> {(analysis_results['anomaly_count'] / analysis_results['total_days'] * 100):.1f}%</p>
            <p><strong>Mean Daily Return:</strong> {analysis_results['mean_return']*100:.2f}%</p>
            <p><strong>Standard Deviation:</strong> {analysis_results['std_return']*100:.2f}%</p>
            <p><strong>Z-Score Threshold:</strong> ±{analysis_results['threshold']}</p>
        </div>
    </div>
    
    <h2>Daily Trading Data</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Close Price</th>
            <th>Daily Return (%)</th>
            <th>Z-Score</th>
            <th>Status</th>
        </tr>
"""
    
    # Add table rows
    for date, row in data.iterrows():
        if pd.isna(row['Daily_Return']):
            continue
            
        is_anomaly = row['Is_Anomaly']
        row_class = 'anomaly' if is_anomaly else ''
        return_class = 'positive' if row['Daily_Return'] > 0 else 'negative'
        status = 'ANOMALY' if is_anomaly else 'Normal'
        
        html_content += f"""
        <tr class="{row_class}">
            <td>{date.strftime('%Y-%m-%d')}</td>
            <td>${row['Close']:.2f}</td>
            <td class="{return_class}">{row['Daily_Return']*100:.2f}%</td>
            <td>{row['Z_Score']:.2f}</td>
            <td>{status}</td>
        </tr>
"""
    
    html_content += """
    </table>
    
    <h2>Anomalous Days Details</h2>
"""
    
    if len(anomalous_days) > 0:
        html_content += "<ul>"
        for date, row in anomalous_days.iterrows():
            direction = "positive" if row['Daily_Return'] > 0 else "negative"
            html_content += f"""
            <li><strong>{date.strftime('%Y-%m-%d')}:</strong> 
                {direction.capitalize()} return of {row['Daily_Return']*100:.2f}% 
                (Z-score: {row['Z_Score']:.2f})
            </li>
"""
        html_content += "</ul>"
    else:
        html_content += "<p>No anomalous days detected in the analysis period.</p>"
    
    html_content += """
    
    <div class="summary">
        <h2>Methodology</h2>
        <p>This analysis uses Z-score based anomaly detection:</p>
        <ul>
            <li>Daily returns are calculated as percentage change in closing price</li>
            <li>Z-scores are calculated as: (return - mean) / standard_deviation</li>
            <li>Days with |Z-score| > 2.0 are flagged as anomalous</li>
            <li>This corresponds to returns more than 2 standard deviations from the mean</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    with open('anomaly_report.html', 'w') as f:
        f.write(html_content)


def main():
    """
    Main function to execute the stock anomaly detection analysis.
    """
    ticker = 'META'
    
    try:
        print(f"Fetching {ticker} stock data for the last 14 trading days...")
        stock_data = fetch_stock_data(ticker, days=14)
        print(f"Successfully fetched {len(stock_data)} days of data")
        
        print("Calculating anomalies using Z-score analysis...")
        analysis_results = calculate_anomalies(stock_data, threshold=2.0)
        print(f"Analysis complete. Found {analysis_results['anomaly_count']} anomalous days")
        
        print("Creating visualization...")
        create_visualization(analysis_results, ticker)
        print("Chart saved to 'anomalies.png'")
        
        print("Generating JSON report...")
        create_json_report(analysis_results, ticker)
        print("JSON report saved to 'anomaly_report.json'")
        
        print("Generating HTML report...")
        create_html_report(analysis_results, ticker)
        print("HTML report saved to 'anomaly_report.html'")
        
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Ticker: {ticker}")
        print(f"Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total trading days: {analysis_results['total_days']}")
        print(f"Anomalous days: {analysis_results['anomaly_count']}")
        print(f"Anomaly rate: {(analysis_results['anomaly_count'] / analysis_results['total_days'] * 100):.1f}%")
        print(f"Mean daily return: {analysis_results['mean_return']*100:.2f}%")
        print(f"Standard deviation: {analysis_results['std_return']*100:.2f}%")
        
        if analysis_results['anomaly_count'] > 0:
            print("\nAnomalous days detected:")
            for date, row in analysis_results['anomalous_days'].iterrows():
                print(f"  {date.strftime('%Y-%m-%d')}: {row['Daily_Return']*100:.2f}% (Z-score: {row['Z_Score']:.2f})")
        else:
            print("\nNo anomalous days detected.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
