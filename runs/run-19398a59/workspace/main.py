#!/usr/bin/env python3
"""
Amazon Stock Anomaly Detection

This script pulls AMZN stock data for the last 14 days,
detects anomalous trading days using Z-score analysis,
and generates reports in multiple formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import logging
import sys
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockAnomalyDetector:
    """
    Detects anomalies in stock trading data using Z-score analysis.
    """
    
    def __init__(self, symbol: str = "AMZN", days: int = 14, z_threshold: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to analyze
            z_threshold: Z-score threshold for anomaly detection
        """
        self.symbol = symbol
        self.days = days
        self.z_threshold = z_threshold
        self.data = None
        self.anomalies = []
        
    def fetch_stock_data(self) -> bool:
        """
        Fetch stock data from yfinance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching {self.symbol} data for last {self.days} days...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Extra days for weekends
            
            # Fetch data
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if self.data.empty:
                logger.error("No data retrieved from yfinance")
                return False
                
            # Keep only the last 14 trading days
            self.data = self.data.tail(self.days)
            
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove first row with NaN return
            self.data = self.data.dropna()
            
            logger.info(f"Successfully fetched {len(self.data)} trading days")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return False
    
    def detect_anomalies(self) -> List[Dict]:
        """
        Detect anomalous trading days using Z-score analysis.
        
        Returns:
            List of dictionaries containing anomaly information
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for anomaly detection")
                return []
            
            logger.info("Detecting anomalies using Z-score analysis...")
            
            # Calculate Z-scores for daily returns
            returns = self.data['Daily_Return']
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                logger.warning("Standard deviation is zero, no anomalies can be detected")
                return []
            
            self.data['Z_Score'] = (returns - mean_return) / std_return
            
            # Identify anomalies
            anomaly_mask = (abs(self.data['Z_Score']) > self.z_threshold)
            anomaly_data = self.data[anomaly_mask]
            
            self.anomalies = []
            for date, row in anomaly_data.iterrows():
                anomaly = {
                    'date': date.strftime('%Y-%m-%d'),
                    'close_price': round(row['Close'], 2),
                    'daily_return': round(row['Daily_Return'] * 100, 2),  # Convert to percentage
                    'z_score': round(row['Z_Score'], 2),
                    'volume': int(row['Volume']),
                    'anomaly_type': 'positive' if row['Z_Score'] > 0 else 'negative'
                }
                self.anomalies.append(anomaly)
            
            logger.info(f"Detected {len(self.anomalies)} anomalous trading days")
            return self.anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def generate_chart(self, filename: str = "anomalies.png") -> bool:
        """
        Generate and save a matplotlib chart highlighting anomalous days.
        
        Args:
            filename: Output filename for the chart
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for chart generation")
                return False
            
            logger.info(f"Generating chart: {filename}")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=2, label='Close Price')
            
            # Highlight anomalous days
            anomaly_dates = [datetime.strptime(a['date'], '%Y-%m-%d') for a in self.anomalies]
            if anomaly_dates:
                anomaly_prices = [self.data.loc[self.data.index.date == d.date(), 'Close'].iloc[0] 
                                for d in anomaly_dates if any(self.data.index.date == d.date())]
                ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=100, 
                           label=f'Anomalies ({len(self.anomalies)})', zorder=5)
            
            ax1.set_title(f'{self.symbol} Stock Price - Last {self.days} Trading Days')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Daily returns with Z-score threshold lines
            returns_pct = self.data['Daily_Return'] * 100
            ax2.bar(self.data.index, returns_pct, alpha=0.7, 
                   color=['red' if abs(z) > self.z_threshold else 'blue' 
                         for z in self.data['Z_Score']])
            
            # Add threshold lines
            mean_return = self.data['Daily_Return'].mean() * 100
            std_return = self.data['Daily_Return'].std() * 100
            
            ax2.axhline(y=mean_return + self.z_threshold * std_return, 
                       color='red', linestyle='--', alpha=0.7, 
                       label=f'+{self.z_threshold}σ threshold')
            ax2.axhline(y=mean_return - self.z_threshold * std_return, 
                       color='red', linestyle='--', alpha=0.7, 
                       label=f'-{self.z_threshold}σ threshold')
            ax2.axhline(y=mean_return, color='black', linestyle='-', alpha=0.5, label='Mean')
            
            ax2.set_title('Daily Returns with Anomaly Thresholds')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Chart saved successfully: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            return False
    
    def generate_json_report(self, filename: str = "anomaly_report.json") -> bool:
        """
        Generate and save a JSON report of anomalies.
        
        Args:
            filename: Output filename for the JSON report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Generating JSON report: {filename}")
            
            # Calculate summary statistics
            if self.data is not None and not self.data.empty:
                summary_stats = {
                    'mean_return': round(self.data['Daily_Return'].mean() * 100, 2),
                    'std_return': round(self.data['Daily_Return'].std() * 100, 2),
                    'min_return': round(self.data['Daily_Return'].min() * 100, 2),
                    'max_return': round(self.data['Daily_Return'].max() * 100, 2),
                    'total_trading_days': len(self.data)
                }
            else:
                summary_stats = {}
            
            report = {
                'analysis_metadata': {
                    'symbol': self.symbol,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'period_days': self.days,
                    'z_score_threshold': self.z_threshold,
                    'total_anomalies': len(self.anomalies)
                },
                'summary_statistics': summary_stats,
                'anomalous_days': self.anomalies
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"JSON report saved successfully: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return False
    
    def generate_html_report(self, filename: str = "anomaly_report.html") -> bool:
        """
        Generate and save an HTML summary report.
        
        Args:
            filename: Output filename for the HTML report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Generating HTML report: {filename}")
            
            # Calculate summary statistics
            if self.data is not None and not self.data.empty:
                mean_return = self.data['Daily_Return'].mean() * 100
                std_return = self.data['Daily_Return'].std() * 100
                min_return = self.data['Daily_Return'].min() * 100
                max_return = self.data['Daily_Return'].max() * 100
                total_days = len(self.data)
            else:
                mean_return = std_return = min_return = max_return = total_days = 0
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.symbol} Stock Anomaly Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
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
        .summary-grid {{
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
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .anomaly-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .anomaly-table th, .anomaly-table td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        .anomaly-table th {{
            background-color: #3498db;
            color: white;
        }}
        .anomaly-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .positive-anomaly {{
            background-color: #d5f4e6;
        }}
        .negative-anomaly {{
            background-color: #ffeaa7;
        }}
        .no-anomalies {{
            text-align: center;
            color: #27ae60;
            font-size: 18px;
            padding: 20px;
            background-color: #d5f4e6;
            border-radius: 8px;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.symbol} Stock Anomaly Analysis Report</h1>
        
        <div class="metadata">
            <h2>Analysis Metadata</h2>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Period:</strong> Last {self.days} trading days</p>
            <p><strong>Z-Score Threshold:</strong> ±{self.z_threshold}</p>
            <p><strong>Total Anomalies Detected:</strong> {len(self.anomalies)}</p>
        </div>
        
        <h2>Summary Statistics</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{total_days}</div>
                <div class="stat-label">Trading Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{mean_return:.2f}%</div>
                <div class="stat-label">Mean Daily Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{std_return:.2f}%</div>
                <div class="stat-label">Return Volatility</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{min_return:.2f}%</div>
                <div class="stat-label">Minimum Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max_return:.2f}%</div>
                <div class="stat-label">Maximum Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.anomalies)}</div>
                <div class="stat-label">Anomalous Days</div>
            </div>
        </div>
        
        <h2>Anomalous Trading Days</h2>
"""
            
            if self.anomalies:
                html_content += """
        <table class="anomaly-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Volume</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
"""
                
                for anomaly in self.anomalies:
                    row_class = "positive-anomaly" if anomaly['anomaly_type'] == 'positive' else "negative-anomaly"
                    html_content += f"""
                <tr class="{row_class}">
                    <td>{anomaly['date']}</td>
                    <td>${anomaly['close_price']}</td>
                    <td>{anomaly['daily_return']:+.2f}%</td>
                    <td>{anomaly['z_score']:+.2f}</td>
                    <td>{anomaly['volume']:,}</td>
                    <td>{anomaly['anomaly_type'].title()}</td>
                </tr>
"""
                
                html_content += """
            </tbody>
        </table>
"""
            else:
                html_content += """
        <div class="no-anomalies">
            🎉 No anomalous trading days detected in the analyzed period!
        </div>
"""
            
            html_content += """
        
        <h2>Methodology</h2>
        <p>This analysis uses Z-score normalization to identify anomalous trading days. A day is considered anomalous if its daily return Z-score exceeds the threshold of ±{threshold}.</p>
        <p><strong>Z-Score Formula:</strong> Z = (Return - Mean) / Standard Deviation</p>
        <p>Where the mean and standard deviation are calculated from all daily returns in the analyzed period.</p>
        
    </div>
</body>
</html>
""".format(threshold=self.z_threshold)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved successfully: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return False

def main():
    """
    Main function to run the stock anomaly detection analysis.
    """
    try:
        logger.info("Starting Amazon Stock Anomaly Detection Analysis")
        
        # Initialize detector
        detector = StockAnomalyDetector(symbol="AMZN", days=14, z_threshold=2.0)
        
        # Fetch stock data
        if not detector.fetch_stock_data():
            logger.error("Failed to fetch stock data. Exiting.")
            sys.exit(1)
        
        # Detect anomalies
        anomalies = detector.detect_anomalies()
        
        # Generate outputs
        success_chart = detector.generate_chart("anomalies.png")
        success_json = detector.generate_json_report("anomaly_report.json")
        success_html = detector.generate_html_report("anomaly_report.html")
        
        # Summary
        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Stock Symbol: {detector.symbol}")
        logger.info(f"Analysis Period: {detector.days} trading days")
        logger.info(f"Z-Score Threshold: ±{detector.z_threshold}")
        logger.info(f"Anomalies Detected: {len(anomalies)}")
        
        if anomalies:
            logger.info("\nAnomalous Days:")
            for anomaly in anomalies:
                logger.info(f"  {anomaly['date']}: {anomaly['daily_return']:+.2f}% (Z={anomaly['z_score']:+.2f})")
        else:
            logger.info("No anomalous trading days detected.")
        
        logger.info("\nGenerated Files:")
        if success_chart:
            logger.info("  ✓ anomalies.png - Visualization chart")
        if success_json:
            logger.info("  ✓ anomaly_report.json - JSON report")
        if success_html:
            logger.info("  ✓ anomaly_report.html - HTML summary report")
        
        logger.info("Analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
