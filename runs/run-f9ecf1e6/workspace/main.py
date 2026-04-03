#!/usr/bin/env python3
"""
META Stock Anomaly Detection

This script pulls META stock data from yfinance for the last 14 days,
detects anomalous trading days using Z-score analysis, and generates
a comprehensive HTML report with visualizations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaStockAnalyzer:
    """
    A class to analyze META stock data and detect anomalous trading days.
    """
    
    def __init__(self, symbol: str = "META", days: int = 14):
        """
        Initialize the analyzer with stock symbol and analysis period.
        
        Args:
            symbol: Stock symbol to analyze (default: META)
            days: Number of days to analyze (default: 14)
        """
        self.symbol = symbol
        self.days = days
        self.data = None
        self.anomalies = []
        
    def fetch_stock_data(self) -> bool:
        """
        Fetch stock data from yfinance for the specified period.
        
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            logger.info(f"Fetching {self.symbol} stock data for last {self.days} days...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Add buffer for weekends
            
            # Fetch data from yfinance
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if self.data.empty:
                logger.error(f"No data retrieved for {self.symbol}")
                return False
                
            # Keep only the last 14 trading days
            self.data = self.data.tail(self.days)
            
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove NaN values
            self.data = self.data.dropna()
            
            logger.info(f"Successfully fetched {len(self.data)} trading days of data")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return False
    
    def detect_anomalies(self, z_threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalous trading days using Z-score analysis.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection (default: 2.0)
            
        Returns:
            List of dictionaries containing anomaly information
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for anomaly detection")
                return []
            
            logger.info(f"Detecting anomalies with Z-score threshold: ±{z_threshold}")
            
            # Calculate Z-scores for daily returns
            returns = self.data['Daily_Return']
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                logger.warning("Standard deviation is zero, no anomalies can be detected")
                return []
            
            self.data['Z_Score'] = (returns - mean_return) / std_return
            
            # Identify anomalies
            anomaly_mask = (abs(self.data['Z_Score']) > z_threshold)
            anomaly_data = self.data[anomaly_mask]
            
            self.anomalies = []
            for date, row in anomaly_data.iterrows():
                anomaly_info = {
                    'date': date.strftime('%Y-%m-%d'),
                    'daily_return': float(row['Daily_Return']),
                    'z_score': float(row['Z_Score']),
                    'close_price': float(row['Close']),
                    'volume': int(row['Volume']),
                    'anomaly_type': 'positive' if row['Z_Score'] > 0 else 'negative'
                }
                self.anomalies.append(anomaly_info)
            
            logger.info(f"Detected {len(self.anomalies)} anomalous trading days")
            return self.anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def create_visualization(self, output_file: str = "anomalies.png") -> bool:
        """
        Create and save a matplotlib visualization highlighting anomalous days.
        
        Args:
            output_file: Output filename for the chart (default: anomalies.png)
            
        Returns:
            bool: True if chart saved successfully, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for visualization")
                return False
            
            logger.info(f"Creating visualization: {output_file}")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{self.symbol} Stock Analysis - Last {self.days} Trading Days', fontsize=16, fontweight='bold')
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=2, label='Close Price')
            
            # Highlight anomalous days
            if self.anomalies:
                anomaly_dates = [datetime.strptime(a['date'], '%Y-%m-%d') for a in self.anomalies]
                anomaly_prices = [self.data.loc[self.data.index.date == d.date(), 'Close'].iloc[0] 
                                for d in anomaly_dates if any(self.data.index.date == d.date())]
                
                if anomaly_dates and anomaly_prices:
                    ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=100, 
                              zorder=5, label='Anomalous Days', marker='o')
            
            ax1.set_title('Stock Price with Anomalous Days Highlighted')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            
            # Plot 2: Daily returns with Z-score threshold lines
            ax2.bar(self.data.index, self.data['Daily_Return'] * 100, 
                   color=['red' if abs(z) > 2 else 'blue' for z in self.data['Z_Score']], 
                   alpha=0.7, label='Daily Returns')
            
            # Add Z-score threshold lines (converted to return percentage)
            mean_return = self.data['Daily_Return'].mean()
            std_return = self.data['Daily_Return'].std()
            upper_threshold = (mean_return + 2 * std_return) * 100
            lower_threshold = (mean_return - 2 * std_return) * 100
            
            ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label='Z-score = +2')
            ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, label='Z-score = -2')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax2.set_title('Daily Returns with Anomaly Thresholds')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            
            # Rotate x-axis labels for better readability
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return False
    
    def generate_json_report(self, output_file: str = "anomaly_report.json") -> bool:
        """
        Generate and save a JSON report with anomaly analysis results.
        
        Args:
            output_file: Output filename for the JSON report (default: anomaly_report.json)
            
        Returns:
            bool: True if report saved successfully, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for JSON report generation")
                return False
            
            logger.info(f"Generating JSON report: {output_file}")
            
            # Calculate summary statistics
            returns = self.data['Daily_Return']
            
            report = {
                'analysis_metadata': {
                    'symbol': self.symbol,
                    'analysis_period_days': self.days,
                    'trading_days_analyzed': len(self.data),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'date_range': {
                        'start': self.data.index[0].strftime('%Y-%m-%d'),
                        'end': self.data.index[-1].strftime('%Y-%m-%d')
                    }
                },
                'summary_statistics': {
                    'mean_daily_return': float(returns.mean()),
                    'std_daily_return': float(returns.std()),
                    'min_daily_return': float(returns.min()),
                    'max_daily_return': float(returns.max()),
                    'total_anomalies_detected': len(self.anomalies)
                },
                'anomalous_days': self.anomalies,
                'all_trading_days': []
            }
            
            # Add all trading days data
            for date, row in self.data.iterrows():
                day_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'daily_return': float(row['Daily_Return']),
                    'z_score': float(row['Z_Score']),
                    'is_anomaly': abs(row['Z_Score']) > 2
                }
                report['all_trading_days'].append(day_data)
            
            # Save JSON report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"JSON report saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return False
    
    def generate_html_report(self, output_file: str = "anomaly_report.html") -> bool:
        """
        Generate and save an HTML summary report.
        
        Args:
            output_file: Output filename for the HTML report (default: anomaly_report.html)
            
        Returns:
            bool: True if report saved successfully, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for HTML report generation")
                return False
            
            logger.info(f"Generating HTML report: {output_file}")
            
            # Calculate summary statistics
            returns = self.data['Daily_Return']
            
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
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
        }}
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .anomaly-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .anomaly-table th, .anomaly-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .anomaly-table th {{
            background-color: #3498db;
            color: white;
        }}
        .anomaly-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .positive-anomaly {{
            background-color: #d4edda !important;
            color: #155724;
        }}
        .negative-anomaly {{
            background-color: #f8d7da !important;
            color: #721c24;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .no-anomalies {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.1em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.symbol} Stock Anomaly Analysis Report</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Analysis Period</h3>
                <div class="value">{self.days} Days</div>
            </div>
            <div class="summary-card">
                <h3>Trading Days</h3>
                <div class="value">{len(self.data)}</div>
            </div>
            <div class="summary-card">
                <h3>Anomalies Detected</h3>
                <div class="value">{len(self.anomalies)}</div>
            </div>
            <div class="summary-card">
                <h3>Mean Daily Return</h3>
                <div class="value">{returns.mean():.2%}</div>
            </div>
        </div>
        
        <h2>Analysis Summary</h2>
        <p><strong>Date Range:</strong> {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Standard Deviation of Returns:</strong> {returns.std():.2%}</p>
        <p><strong>Min Daily Return:</strong> {returns.min():.2%}</p>
        <p><strong>Max Daily Return:</strong> {returns.max():.2%}</p>
        
        <h2>Anomalous Trading Days</h2>
"""
            
            if self.anomalies:
                html_content += """
        <p>The following trading days were identified as anomalous (Z-score > 2 or < -2):</p>
        <table class="anomaly-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Daily Return</th>
                    <th>Z-Score</th>
                    <th>Close Price</th>
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
                    <td>{anomaly['daily_return']:.2%}</td>
                    <td>{anomaly['z_score']:.2f}</td>
                    <td>${anomaly['close_price']:.2f}</td>
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
            <p>No anomalous trading days detected during the analysis period.</p>
            <p>All daily returns had Z-scores within the ±2 threshold.</p>
        </div>
"""
            
            html_content += f"""
        
        <h2>Visualization</h2>
        <div class="chart-container">
            <img src="anomalies.png" alt="Stock Analysis Chart" />
            <p><em>Chart showing stock price and daily returns with anomalous days highlighted</em></p>
        </div>
        
        <h2>Methodology</h2>
        <p>This analysis uses Z-score normalization to identify anomalous trading days:</p>
        <ul>
            <li><strong>Z-score calculation:</strong> (Daily Return - Mean Return) / Standard Deviation</li>
            <li><strong>Anomaly threshold:</strong> |Z-score| > 2 (approximately 95% confidence interval)</li>
            <li><strong>Data source:</strong> Yahoo Finance via yfinance library</li>
            <li><strong>Analysis period:</strong> Last {self.days} trading days</li>
        </ul>
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>Data provided by Yahoo Finance. This analysis is for informational purposes only.</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return False

def main():
    """
    Main function to execute the META stock anomaly analysis.
    """
    try:
        logger.info("Starting META stock anomaly analysis...")
        
        # Initialize analyzer
        analyzer = MetaStockAnalyzer(symbol="META", days=14)
        
        # Fetch stock data
        if not analyzer.fetch_stock_data():
            logger.error("Failed to fetch stock data. Exiting.")
            return False
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(z_threshold=2.0)
        
        # Generate outputs
        success_chart = analyzer.create_visualization("anomalies.png")
        success_json = analyzer.generate_json_report("anomaly_report.json")
        success_html = analyzer.generate_html_report("anomaly_report.html")
        
        # Summary
        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Stock Symbol: {analyzer.symbol}")
        logger.info(f"Trading Days Analyzed: {len(analyzer.data)}")
        logger.info(f"Anomalies Detected: {len(anomalies)}")
        
        if anomalies:
            logger.info("Anomalous Days:")
            for anomaly in anomalies:
                logger.info(f"  - {anomaly['date']}: {anomaly['daily_return']:.2%} (Z-score: {anomaly['z_score']:.2f})")
        else:
            logger.info("No anomalous trading days detected.")
        
        logger.info("\nOutput Files:")
        logger.info(f"  - Chart: anomalies.png {'✓' if success_chart else '✗'}")
        logger.info(f"  - JSON Report: anomaly_report.json {'✓' if success_json else '✗'}")
        logger.info(f"  - HTML Report: anomaly_report.html {'✓' if success_html else '✗'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
