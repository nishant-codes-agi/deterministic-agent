#!/usr/bin/env python3
"""
AMZN Stock Anomaly Detection

This script fetches AMZN stock data for the last 14 days,
detects anomalous trading days using Z-score analysis,
and generates comprehensive reports in multiple formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import sys
from typing import Dict, List, Tuple, Optional


class StockAnomalyDetector:
    """Class to handle stock data fetching and anomaly detection."""
    
    def __init__(self, ticker: str = "AMZN", days: int = 14, z_threshold: float = 2.0):
        self.ticker = ticker
        self.days = days
        self.z_threshold = z_threshold
        self.data = None
        self.anomalies = None
        
    def fetch_stock_data(self) -> bool:
        """
        Fetch historical stock data using yfinance.
        
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            
            print(f"Fetching {self.ticker} data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data with error handling
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date, interval="1d")
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
                
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_anomalies(self) -> bool:
        """
        Calculate daily returns and detect anomalies using Z-score.
        
        Returns:
            bool: True if calculation successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for anomaly calculation")
                return False
            
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove NaN values (first day has no previous day for return calculation)
            returns = self.data['Daily_Return'].dropna()
            
            if len(returns) < 2:
                print("Error: Insufficient data for statistical analysis")
                return False
            
            # Calculate statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                print("Warning: Standard deviation is zero, no anomalies can be detected")
                return False
            
            # Calculate Z-scores
            self.data['Z_Score'] = (self.data['Daily_Return'] - mean_return) / std_return
            
            # Identify anomalies
            anomaly_mask = (abs(self.data['Z_Score']) > self.z_threshold) & (~self.data['Daily_Return'].isna())
            self.anomalies = self.data[anomaly_mask].copy()
            
            print(f"Detected {len(self.anomalies)} anomalous trading days")
            return True
            
        except Exception as e:
            print(f"Error calculating anomalies: {str(e)}")
            return False
    
    def generate_html_report(self, filename: str = "stock_report.html") -> bool:
        """
        Generate HTML report with anomaly analysis.
        
        Args:
            filename: Output HTML filename
            
        Returns:
            bool: True if report generated successfully, False otherwise
        """
        try:
            if self.data is None or self.anomalies is None:
                print("Error: No data available for HTML report generation")
                return False
            
            # Calculate summary statistics
            returns = self.data['Daily_Return'].dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            min_return = returns.min()
            max_return = returns.max()
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.ticker} Stock Anomaly Report</title>
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
        .stat {{
            display: inline-block;
            margin: 10px 20px;
            padding: 10px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .anomaly-positive {{
            background-color: #d4edda !important;
            color: #155724;
        }}
        .anomaly-negative {{
            background-color: #f8d7da !important;
            color: #721c24;
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
        <h1>{self.ticker} Stock Anomaly Detection Report</h1>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p><strong>Analysis Period:</strong> {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}</p>
            <p><strong>Total Trading Days:</strong> {len(self.data)}</p>
            <p><strong>Anomalous Days Detected:</strong> {len(self.anomalies)}</p>
            <p><strong>Z-Score Threshold:</strong> ±{self.z_threshold}</p>
        </div>
        
        <div class="summary">
            <h2>Statistical Overview</h2>
            <div class="stat">Mean Daily Return: {mean_return:.4f} ({mean_return*100:.2f}%)</div>
            <div class="stat">Std Dev: {std_return:.4f} ({std_return*100:.2f}%)</div>
            <div class="stat">Min Return: {min_return:.4f} ({min_return*100:.2f}%)</div>
            <div class="stat">Max Return: {max_return:.4f} ({max_return*100:.2f}%)</div>
        </div>
"""
            
            if len(self.anomalies) > 0:
                html_content += """
        <h2>Anomalous Trading Days</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Anomaly Type</th>
                </tr>
            </thead>
            <tbody>
"""
                
                for date, row in self.anomalies.iterrows():
                    anomaly_type = "Positive" if row['Z_Score'] > 0 else "Negative"
                    css_class = "anomaly-positive" if row['Z_Score'] > 0 else "anomaly-negative"
                    
                    html_content += f"""
                <tr class="{css_class}">
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>${row['Close']:.2f}</td>
                    <td>{row['Daily_Return']*100:.2f}%</td>
                    <td>{row['Z_Score']:.2f}</td>
                    <td>{anomaly_type}</td>
                </tr>
"""
                
                html_content += """
            </tbody>
        </table>
"""
            else:
                html_content += """
        <h2>Anomalous Trading Days</h2>
        <p style="text-align: center; color: #27ae60; font-size: 18px;">No anomalous trading days detected in the analysis period.</p>
"""
            
            html_content += f"""
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Data source: Yahoo Finance via yfinance library</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Write HTML file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML report saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            return False
    
    def generate_json_report(self, filename: str = "anomaly_report.json") -> bool:
        """
        Generate JSON report with anomaly data.
        
        Args:
            filename: Output JSON filename
            
        Returns:
            bool: True if report generated successfully, False otherwise
        """
        try:
            if self.data is None or self.anomalies is None:
                print("Error: No data available for JSON report generation")
                return False
            
            # Prepare data for JSON serialization
            returns = self.data['Daily_Return'].dropna()
            
            report_data = {
                "analysis_info": {
                    "ticker": self.ticker,
                    "analysis_period": {
                        "start_date": self.data.index[0].strftime('%Y-%m-%d'),
                        "end_date": self.data.index[-1].strftime('%Y-%m-%d')
                    },
                    "total_trading_days": len(self.data),
                    "z_score_threshold": self.z_threshold,
                    "anomalies_detected": len(self.anomalies)
                },
                "statistics": {
                    "mean_daily_return": float(returns.mean()),
                    "std_daily_return": float(returns.std()),
                    "min_daily_return": float(returns.min()),
                    "max_daily_return": float(returns.max())
                },
                "anomalous_days": []
            }
            
            # Add anomalous days data
            for date, row in self.anomalies.iterrows():
                anomaly_data = {
                    "date": date.strftime('%Y-%m-%d'),
                    "close_price": float(row['Close']),
                    "daily_return": float(row['Daily_Return']),
                    "z_score": float(row['Z_Score']),
                    "anomaly_type": "positive" if row['Z_Score'] > 0 else "negative"
                }
                report_data["anomalous_days"].append(anomaly_data)
            
            # Write JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"JSON report saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating JSON report: {str(e)}")
            return False
    
    def generate_chart(self, filename: str = "anomalies.png") -> bool:
        """
        Generate matplotlib chart highlighting anomalous days.
        
        Args:
            filename: Output PNG filename
            
        Returns:
            bool: True if chart generated successfully, False otherwise
        """
        try:
            if self.data is None:
                print("Error: No data available for chart generation")
                return False
            
            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{self.ticker} Stock Price and Anomaly Detection', fontsize=16, fontweight='bold')
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=2, label='Close Price')
            
            if len(self.anomalies) > 0:
                ax1.scatter(self.anomalies.index, self.anomalies['Close'], 
                           c='red', s=100, alpha=0.7, zorder=5, label='Anomalous Days')
            
            ax1.set_title('Stock Price with Anomalous Days Highlighted')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Daily returns with Z-score threshold lines
            returns_data = self.data['Daily_Return'].dropna()
            ax2.bar(returns_data.index, returns_data * 100, alpha=0.7, color='lightblue', label='Daily Returns')
            
            if len(self.anomalies) > 0:
                anomaly_returns = self.anomalies['Daily_Return'] * 100
                ax2.bar(self.anomalies.index, anomaly_returns, 
                       color='red', alpha=0.8, label='Anomalous Returns')
            
            # Add Z-score threshold lines (converted to percentage)
            if len(returns_data) > 1:
                mean_return = returns_data.mean()
                std_return = returns_data.std()
                upper_threshold = (mean_return + self.z_threshold * std_return) * 100
                lower_threshold = (mean_return - self.z_threshold * std_return) * 100
                
                ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Z-score +{self.z_threshold} threshold')
                ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Z-score -{self.z_threshold} threshold')
            
            ax2.set_title('Daily Returns with Anomaly Thresholds')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            return False


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("AMZN Stock Anomaly Detection Analysis")
    print("=" * 60)
    
    # Initialize detector
    detector = StockAnomalyDetector(ticker="AMZN", days=14, z_threshold=2.0)
    
    # Step 1: Fetch stock data
    print("\n1. Fetching stock data...")
    if not detector.fetch_stock_data():
        print("Failed to fetch stock data. Exiting.")
        sys.exit(1)
    
    # Step 2: Calculate anomalies
    print("\n2. Calculating anomalies...")
    if not detector.calculate_anomalies():
        print("Failed to calculate anomalies. Exiting.")
        sys.exit(1)
    
    # Step 3: Generate reports
    print("\n3. Generating reports...")
    
    success_count = 0
    
    # Generate HTML report
    if detector.generate_html_report():
        success_count += 1
    
    # Generate JSON report
    if detector.generate_json_report():
        success_count += 1
    
    # Generate chart
    if detector.generate_chart():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Reports generated successfully: {success_count}/3")
    
    if detector.anomalies is not None:
        print(f"Anomalous trading days detected: {len(detector.anomalies)}")
        if len(detector.anomalies) > 0:
            print("\nAnomalous dates:")
            for date, row in detector.anomalies.iterrows():
                print(f"  - {date.strftime('%Y-%m-%d')}: {row['Daily_Return']*100:.2f}% (Z-score: {row['Z_Score']:.2f})")
        else:
            print("No anomalous trading days found in the analysis period.")
    
    print("\nOutput files:")
    print("  - stock_report.html (HTML report)")
    print("  - anomaly_report.json (JSON data)")
    print("  - anomalies.png (Chart visualization)")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()