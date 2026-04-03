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
        Fetch stock data from yfinance for the specified period.
        
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            
            print(f"Fetching {self.ticker} data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data using yfinance
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date)
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
                
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_daily_returns(self) -> None:
        """
        Calculate daily returns and Z-scores for the stock data.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No stock data available. Please fetch data first.")
        
        # Calculate daily returns (percentage change)
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # Remove NaN values (first day has no previous day to compare)
        returns = self.data['Daily_Return'].dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns calculated")
        
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            print("Warning: Standard deviation is 0, cannot calculate Z-scores")
            self.data['Z_Score'] = 0
        else:
            # Calculate Z-scores
            self.data['Z_Score'] = (self.data['Daily_Return'] - mean_return) / std_return
        
        print(f"Daily returns calculated. Mean: {mean_return:.4f}, Std: {std_return:.4f}")
    
    def detect_anomalies(self) -> pd.DataFrame:
        """
        Detect anomalous trading days based on Z-score threshold.
        
        Returns:
            pd.DataFrame: DataFrame containing anomalous days
        """
        if self.data is None:
            raise ValueError("No data available for anomaly detection")
        
        # Find anomalies where absolute Z-score > threshold
        anomaly_mask = (abs(self.data['Z_Score']) > self.z_threshold) & (~self.data['Z_Score'].isna())
        self.anomalies = self.data[anomaly_mask].copy()
        
        print(f"Found {len(self.anomalies)} anomalous trading days")
        return self.anomalies
    
    def generate_chart(self, filename: str = "anomalies.png") -> None:
        """
        Generate and save a matplotlib chart highlighting anomalous days.
        
        Args:
            filename (str): Output filename for the chart
        """
        if self.data is None:
            raise ValueError("No data available for chart generation")
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{self.ticker} Stock Analysis - Last {self.days} Days', fontsize=16, fontweight='bold')
        
        # Plot 1: Stock price with anomalous days highlighted
        ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=2, label='Close Price')
        
        if not self.anomalies.empty:
            ax1.scatter(self.anomalies.index, self.anomalies['Close'], 
                       color='red', s=100, zorder=5, label='Anomalous Days')
        
        ax1.set_title('Stock Price with Anomalous Days Highlighted')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Daily returns with Z-score threshold lines
        valid_returns = self.data.dropna(subset=['Daily_Return'])
        ax2.bar(valid_returns.index, valid_returns['Daily_Return'], 
               alpha=0.7, color='lightblue', label='Daily Returns')
        
        if not self.anomalies.empty:
            ax2.bar(self.anomalies.index, self.anomalies['Daily_Return'], 
                   color='red', alpha=0.8, label='Anomalous Returns')
        
        # Add threshold lines
        if len(valid_returns) > 0 and valid_returns['Daily_Return'].std() > 0:
            mean_ret = valid_returns['Daily_Return'].mean()
            std_ret = valid_returns['Daily_Return'].std()
            upper_threshold = mean_ret + self.z_threshold * std_ret
            lower_threshold = mean_ret - self.z_threshold * std_ret
            
            ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score ±{self.z_threshold}')
            ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_title('Daily Returns with Anomaly Thresholds')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Chart saved as {filename}")
    
    def generate_json_report(self, filename: str = "anomaly_report.json") -> None:
        """
        Generate and save a JSON report with anomaly details.
        
        Args:
            filename (str): Output filename for the JSON report
        """
        if self.data is None:
            raise ValueError("No data available for report generation")
        
        # Calculate summary statistics
        valid_returns = self.data['Daily_Return'].dropna()
        
        report = {
            "analysis_metadata": {
                "ticker": self.ticker,
                "analysis_date": datetime.now().isoformat(),
                "period_days": self.days,
                "z_score_threshold": self.z_threshold,
                "total_trading_days": len(self.data),
                "valid_return_days": len(valid_returns)
            },
            "summary_statistics": {
                "mean_daily_return": float(valid_returns.mean()) if len(valid_returns) > 0 else None,
                "std_daily_return": float(valid_returns.std()) if len(valid_returns) > 0 else None,
                "min_daily_return": float(valid_returns.min()) if len(valid_returns) > 0 else None,
                "max_daily_return": float(valid_returns.max()) if len(valid_returns) > 0 else None,
                "total_anomalies_detected": len(self.anomalies) if self.anomalies is not None else 0
            },
            "anomalous_days": []
        }
        
        # Add anomalous days details
        if self.anomalies is not None and not self.anomalies.empty:
            for date, row in self.anomalies.iterrows():
                anomaly_data = {
                    "date": date.strftime('%Y-%m-%d'),
                    "close_price": float(row['Close']),
                    "daily_return": float(row['Daily_Return']) if not pd.isna(row['Daily_Return']) else None,
                    "z_score": float(row['Z_Score']) if not pd.isna(row['Z_Score']) else None,
                    "volume": int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else None
                }
                report["anomalous_days"].append(anomaly_data)
        
        # Save JSON report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved as {filename}")
    
    def generate_html_report(self, filename: str = "stock_report.html") -> None:
        """
        Generate and save an HTML report with analysis results.
        
        Args:
            filename (str): Output filename for the HTML report
        """
        if self.data is None:
            raise ValueError("No data available for report generation")
        
        # Calculate summary statistics
        valid_returns = self.data['Daily_Return'].dropna()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.ticker} Stock Anomaly Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
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
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2980b9;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .anomaly-high {{
            background-color: #ffebee;
        }}
        .anomaly-low {{
            background-color: #e8f5e8;
        }}
        .no-anomalies {{
            text-align: center;
            color: #27ae60;
            font-style: italic;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.ticker} Stock Anomaly Analysis Report</h1>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{len(self.data)}</div>
                <div class="stat-label">Total Trading Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(valid_returns)}</div>
                <div class="stat-label">Valid Return Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{valid_returns.mean():.4f}</div>
                <div class="stat-label">Mean Daily Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{valid_returns.std():.4f}</div>
                <div class="stat-label">Return Std Dev</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.anomalies) if self.anomalies is not None else 0}</div>
                <div class="stat-label">Anomalies Detected</div>
            </div>
        </div>
        
        <h2>Analysis Parameters</h2>
        <ul>
            <li><strong>Ticker:</strong> {self.ticker}</li>
            <li><strong>Analysis Period:</strong> Last {self.days} days</li>
            <li><strong>Z-Score Threshold:</strong> ±{self.z_threshold}</li>
            <li><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        
        <h2>Anomalous Trading Days</h2>
"""
        
        # Add anomalies table
        if self.anomalies is not None and not self.anomalies.empty:
            html_content += """
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Volume</th>
                    <th>Anomaly Type</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for date, row in self.anomalies.iterrows():
                return_pct = row['Daily_Return'] * 100 if not pd.isna(row['Daily_Return']) else 0
                z_score = row['Z_Score'] if not pd.isna(row['Z_Score']) else 0
                
                anomaly_type = "High Volatility" if z_score > self.z_threshold else "Low Volatility"
                row_class = "anomaly-high" if z_score > self.z_threshold else "anomaly-low"
                return_class = "positive" if return_pct > 0 else "negative"
                
                html_content += f"""
                <tr class="{row_class}">
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>${row['Close']:.2f}</td>
                    <td class="{return_class}">{return_pct:.2f}%</td>
                    <td>{z_score:.2f}</td>
                    <td>{int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 'N/A'}</td>
                    <td>{anomaly_type}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
        else:
            html_content += """
        <div class="no-anomalies">
            <p>No anomalous trading days detected in the analyzed period.</p>
        </div>
"""
        
        html_content += """
        
        <h2>Methodology</h2>
        <p>This analysis identifies anomalous trading days using Z-score analysis of daily returns:</p>
        <ul>
            <li><strong>Daily Return:</strong> Calculated as percentage change in closing price from previous day</li>
            <li><strong>Z-Score:</strong> Measures how many standard deviations a return is from the mean</li>
            <li><strong>Anomaly Threshold:</strong> Days with |Z-score| > 2.0 are considered anomalous</li>
            <li><strong>High Volatility:</strong> Z-score > +2.0 (unusually high positive or negative returns)</li>
            <li><strong>Low Volatility:</strong> Z-score < -2.0 (unusually low returns relative to typical volatility)</li>
        </ul>
        
        <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</em></p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved as {filename}")


def main():
    """
    Main execution function that orchestrates the entire analysis process.
    """
    print("=" * 60)
    print("AMZN Stock Anomaly Detection Analysis")
    print("=" * 60)
    
    try:
        # Initialize the detector
        detector = StockAnomalyDetector(ticker="AMZN", days=14, z_threshold=2.0)
        
        # Step 1: Fetch stock data
        print("\n1. Fetching stock data...")
        if not detector.fetch_stock_data():
            print("Failed to fetch stock data. Exiting.")
            sys.exit(1)
        
        # Step 2: Calculate daily returns and Z-scores
        print("\n2. Calculating daily returns and Z-scores...")
        detector.calculate_daily_returns()
        
        # Step 3: Detect anomalies
        print("\n3. Detecting anomalous trading days...")
        anomalies = detector.detect_anomalies()
        
        # Step 4: Generate reports
        print("\n4. Generating reports...")
        
        # Generate chart
        detector.generate_chart("anomalies.png")
        
        # Generate JSON report
        detector.generate_json_report("anomaly_report.json")
        
        # Generate HTML report
        detector.generate_html_report("stock_report.html")
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Total trading days analyzed: {len(detector.data)}")
        print(f"Anomalous days detected: {len(anomalies)}")
        print("\nGenerated files:")
        print("  - anomalies.png (chart)")
        print("  - anomaly_report.json (JSON report)")
        print("  - stock_report.html (HTML report)")
        
        if not anomalies.empty:
            print("\nAnomalous days summary:")
            for date, row in anomalies.iterrows():
                return_pct = row['Daily_Return'] * 100 if not pd.isna(row['Daily_Return']) else 0
                z_score = row['Z_Score'] if not pd.isna(row['Z_Score']) else 0
                print(f"  {date.strftime('%Y-%m-%d')}: {return_pct:+.2f}% return (Z-score: {z_score:.2f})")
        else:
            print("\nNo anomalous trading days detected in the analyzed period.")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
