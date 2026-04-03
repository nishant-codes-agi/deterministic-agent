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
        start_date = end_date - timedelta(days=days + 5)  # Extra days for weekends
        
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Get last 14 trading days
        data = data.tail(days)
        
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
    anomalies = data[data['Is_Anomaly'] == True].copy()
    
    return {
        'data': data,
        'anomalies': anomalies,
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'threshold': threshold,
        'total_days': len(data),
        'anomaly_count': len(anomalies)
    }


def create_visualization(analysis_results: Dict[str, Any], output_file: str = 'anomalies.png') -> None:
    """
    Create and save a matplotlib chart highlighting anomalous days.
    
    Args:
        analysis_results: Results from anomaly analysis
        output_file: Output filename for the chart
    """
    data = analysis_results['data']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Stock price with anomalous days highlighted
    ax1.plot(data.index, data['Close'], 'b-', linewidth=2, label='Close Price')
    
    # Highlight anomalous days
    anomaly_dates = data[data['Is_Anomaly']].index
    anomaly_prices = data[data['Is_Anomaly']]['Close']
    
    if len(anomaly_dates) > 0:
        ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=100, 
                   label=f'Anomalous Days ({len(anomaly_dates)})', zorder=5)
    
    ax1.set_title('META Stock Price - Last 14 Trading Days', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily returns with Z-score threshold lines
    ax2.bar(data.index, data['Daily_Return'] * 100, alpha=0.7, 
           color=['red' if anomaly else 'blue' for anomaly in data['Is_Anomaly']])
    
    # Add threshold lines
    threshold = analysis_results['threshold']
    mean_return = analysis_results['mean_return']
    std_return = analysis_results['std_return']
    
    upper_threshold = (mean_return + threshold * std_return) * 100
    lower_threshold = (mean_return - threshold * std_return) * 100
    
    ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Z-score = +{threshold}')
    ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Z-score = -{threshold}')
    ax2.axhline(y=mean_return * 100, color='green', linestyle='-', alpha=0.7, 
               label='Mean Return')
    
    ax2.set_title('Daily Returns with Anomaly Thresholds', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {output_file}")


def create_json_report(analysis_results: Dict[str, Any], output_file: str = 'anomaly_report.json') -> None:
    """
    Create and save a JSON report with anomaly analysis results.
    
    Args:
        analysis_results: Results from anomaly analysis
        output_file: Output filename for the JSON report
    """
    anomalies = analysis_results['anomalies']
    
    # Prepare anomaly data for JSON serialization
    anomaly_list = []
    for date, row in anomalies.iterrows():
        anomaly_list.append({
            'date': date.strftime('%Y-%m-%d'),
            'close_price': float(row['Close']),
            'daily_return': float(row['Daily_Return']),
            'daily_return_percent': float(row['Daily_Return'] * 100),
            'z_score': float(row['Z_Score'])
        })
    
    # Create comprehensive report
    report = {
        'analysis_summary': {
            'ticker': 'META',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period_days': analysis_results['total_days'],
            'anomaly_threshold': analysis_results['threshold'],
            'total_anomalies': analysis_results['anomaly_count']
        },
        'statistics': {
            'mean_daily_return': analysis_results['mean_return'],
            'mean_daily_return_percent': analysis_results['mean_return'] * 100,
            'std_daily_return': analysis_results['std_return'],
            'std_daily_return_percent': analysis_results['std_return'] * 100
        },
        'anomalous_days': anomaly_list,
        'methodology': {
            'description': 'Anomalies detected using Z-score analysis of daily returns',
            'formula': 'Z-score = (daily_return - mean_return) / std_return',
            'threshold_criteria': f'|Z-score| > {analysis_results["threshold"]}'
        }
    }
    
    # Save JSON report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved to {output_file}")


def create_html_report(analysis_results: Dict[str, Any], output_file: str = 'anomaly_report.html') -> None:
    """
    Create and save an HTML report with anomaly analysis results.
    
    Args:
        analysis_results: Results from anomaly analysis
        output_file: Output filename for the HTML report
    """
    data = analysis_results['data']
    anomalies = analysis_results['anomalies']
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>META Stock Anomaly Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .anomaly-alert {{
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: center;
            font-weight: bold;
        }}
        .no-anomaly {{
            background-color: #27ae60;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: center;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .anomaly-row {{
            background-color: #ffebee;
            font-weight: bold;
        }}
        .positive-return {{
            color: #27ae60;
        }}
        .negative-return {{
            color: #e74c3c;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>META Stock Anomaly Analysis Report</h1>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p><strong>Ticker:</strong> META</p>
            <p><strong>Analysis Period:</strong> Last {analysis_results['total_days']} trading days</p>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Anomaly Threshold:</strong> Z-score > {analysis_results['threshold']} or < -{analysis_results['threshold']}</p>
            <p><strong>Mean Daily Return:</strong> {analysis_results['mean_return']*100:.4f}%</p>
            <p><strong>Standard Deviation:</strong> {analysis_results['std_return']*100:.4f}%</p>
        </div>
"""
    
    # Add anomaly alert or success message
    if analysis_results['anomaly_count'] > 0:
        html_content += f"""
        <div class="anomaly-alert">
            ⚠️ {analysis_results['anomaly_count']} anomalous trading day(s) detected!
        </div>
"""
    else:
        html_content += """
        <div class="no-anomaly">
            ✅ No anomalous trading days detected in the analysis period.
        </div>
"""
    
    # Add detailed table
    html_content += """
        <h2>Detailed Trading Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Anomaly</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add table rows
    for date, row in data.iterrows():
        if pd.isna(row['Daily_Return']):
            continue
            
        is_anomaly = row['Is_Anomaly']
        row_class = 'anomaly-row' if is_anomaly else ''
        return_class = 'positive-return' if row['Daily_Return'] > 0 else 'negative-return'
        anomaly_text = '🚨 YES' if is_anomaly else 'No'
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>${float(row['Close']):.2f}</td>
                    <td class="{return_class}">{float(row['Daily_Return'])*100:.4f}%</td>
                    <td>{float(row['Z_Score']):.4f}</td>
                    <td>{anomaly_text}</td>
                </tr>
"""
    
    # Close HTML
    html_content += """
            </tbody>
        </table>
        
        <h2>Methodology</h2>
        <div class="summary">
            <p><strong>Anomaly Detection Method:</strong> Z-score analysis of daily returns</p>
            <p><strong>Formula:</strong> Z-score = (daily_return - mean_return) / std_deviation</p>
            <p><strong>Threshold:</strong> Days with |Z-score| > 2.0 are considered anomalous</p>
            <p><strong>Daily Return Calculation:</strong> (Close_today - Close_yesterday) / Close_yesterday</p>
        </div>
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Data source: Yahoo Finance via yfinance library</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_file}")


def main():
    """
    Main function to execute the anomaly detection analysis.
    """
    try:
        print("Starting META stock anomaly analysis...")
        print("=" * 50)
        
        # Fetch stock data
        print("1. Fetching META stock data for the last 14 trading days...")
        stock_data = fetch_stock_data('META', 14)
        print(f"   ✓ Successfully fetched {len(stock_data)} trading days of data")
        
        # Calculate anomalies
        print("\n2. Calculating daily returns and detecting anomalies...")
        analysis_results = calculate_anomalies(stock_data, threshold=2.0)
        print(f"   ✓ Analysis complete")
        print(f"   ✓ Mean daily return: {analysis_results['mean_return']*100:.4f}%")
        print(f"   ✓ Standard deviation: {analysis_results['std_return']*100:.4f}%")
        print(f"   ✓ Anomalies detected: {analysis_results['anomaly_count']}")
        
        # Generate reports
        print("\n3. Generating reports...")
        
        # Create visualization
        create_visualization(analysis_results)
        
        # Create JSON report
        create_json_report(analysis_results)
        
        # Create HTML report
        create_html_report(analysis_results)
        
        print("\n" + "=" * 50)
        print("Analysis complete! Generated files:")
        print("  • anomalies.png - Visualization chart")
        print("  • anomaly_report.json - Detailed JSON report")
        print("  • anomaly_report.html - Interactive HTML report")
        
        # Print summary
        if analysis_results['anomaly_count'] > 0:
            print(f"\n⚠️  {analysis_results['anomaly_count']} anomalous trading day(s) detected!")
            print("\nAnomalous days:")
            for date, row in analysis_results['anomalies'].iterrows():
                print(f"  • {date.strftime('%Y-%m-%d')}: {float(row['Daily_Return'])*100:.4f}% return (Z-score: {float(row['Z_Score']):.4f})")
        else:
            print("\n✅ No anomalous trading days detected.")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  • Check your internet connection")
        print("  • Verify that META is a valid ticker symbol")
        print("  • Ensure you have the required libraries installed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
