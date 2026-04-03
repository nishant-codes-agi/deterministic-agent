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
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class StockAnomalyDetector:
    """Detects anomalies in stock trading data using Z-score analysis."""
    
    def __init__(self, ticker='META', days=14, z_threshold=2.0):
        self.ticker = ticker
        self.days = days
        self.z_threshold = z_threshold
        self.data = None
        self.anomalies = None
    
    def fetch_stock_data(self):
        """Fetch historical stock data from yfinance."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Extra days to ensure we get enough trading days
            
            print(f"Fetching {self.ticker} stock data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Keep only the last 14 trading days
            data = data.tail(self.days)
            
            if len(data) < 5:
                raise ValueError(f"Insufficient data: only {len(data)} trading days available")
            
            self.data = data
            print(f"Successfully fetched {len(data)} trading days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_anomalies(self):
        """Calculate daily returns and detect anomalies using Z-score."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data() first.")
        
        # Calculate daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # Remove NaN values (first day has no previous day for return calculation)
        returns = self.data['Daily_Return'].dropna()
        
        if len(returns) < 2:
            raise ValueError("Insufficient data for anomaly detection")
        
        # Calculate Z-scores
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            print("Warning: Standard deviation is 0, no anomalies can be detected")
            self.data['Z_Score'] = 0
            self.data['Is_Anomaly'] = False
        else:
            self.data['Z_Score'] = (self.data['Daily_Return'] - mean_return) / std_return
            self.data['Is_Anomaly'] = abs(self.data['Z_Score']) > self.z_threshold
        
        # Filter anomalies
        self.anomalies = self.data[self.data['Is_Anomaly'] == True].copy()
        
        print(f"Detected {len(self.anomalies)} anomalous trading days")
        print(f"Mean daily return: {mean_return:.4f} ({mean_return*100:.2f}%)")
        print(f"Standard deviation: {std_return:.4f} ({std_return*100:.2f}%)")
    
    def generate_chart(self, filename='anomalies.png'):
        """Generate and save a matplotlib chart highlighting anomalous days."""
        if self.data is None:
            raise ValueError("No data available for chart generation")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Stock price with anomalous days highlighted
        dates = self.data.index
        prices = self.data['Close']
        
        ax1.plot(dates, prices, 'b-', linewidth=2, label='Close Price')
        
        # Highlight anomalous days
        if len(self.anomalies) > 0:
            anomaly_dates = self.anomalies.index
            anomaly_prices = self.anomalies['Close']
            ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=100, 
                       label=f'Anomalous Days ({len(self.anomalies)})', zorder=5)
        
        ax1.set_title(f'{self.ticker} Stock Price - Last {self.days} Trading Days', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        # Plot 2: Daily returns with Z-score threshold lines
        returns = self.data['Daily_Return'] * 100  # Convert to percentage
        z_scores = self.data['Z_Score']
        
        # Create bars for returns
        colors = ['red' if anomaly else 'blue' for anomaly in self.data['Is_Anomaly']]
        ax2.bar(dates, returns, color=colors, alpha=0.7, label='Daily Returns')
        
        # Add Z-score threshold lines (converted to return percentage)
        if self.data['Daily_Return'].std() > 0:
            mean_return = self.data['Daily_Return'].mean() * 100
            std_return = self.data['Daily_Return'].std() * 100
            upper_threshold = mean_return + (self.z_threshold * std_return)
            lower_threshold = mean_return - (self.z_threshold * std_return)
            
            ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.8, 
                       label=f'Z-score = +{self.z_threshold}')
            ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.8, 
                       label=f'Z-score = -{self.z_threshold}')
            ax2.axhline(y=mean_return, color='green', linestyle='-', alpha=0.8, 
                       label='Mean Return')
        
        ax2.set_title('Daily Returns with Anomaly Thresholds', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Daily Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Chart saved as {filename}")
    
    def generate_json_report(self, filename='anomaly_report.json'):
        """Generate and save a JSON report with anomalous dates and returns."""
        if self.data is None:
            raise ValueError("No data available for report generation")
        
        # Prepare report data
        report = {
            'ticker': self.ticker,
            'analysis_period': {
                'start_date': self.data.index[0].strftime('%Y-%m-%d'),
                'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
                'trading_days': len(self.data)
            },
            'parameters': {
                'z_score_threshold': self.z_threshold
            },
            'statistics': {
                'mean_daily_return': float(self.data['Daily_Return'].mean()),
                'std_daily_return': float(self.data['Daily_Return'].std()),
                'total_anomalies': len(self.anomalies)
            },
            'anomalous_days': []
        }
        
        # Add anomalous days details
        for date, row in self.anomalies.iterrows():
            anomaly_data = {
                'date': date.strftime('%Y-%m-%d'),
                'close_price': float(row['Close']),
                'daily_return': float(row['Daily_Return']),
                'daily_return_percent': float(row['Daily_Return'] * 100),
                'z_score': float(row['Z_Score']),
                'volume': int(row['Volume']) if 'Volume' in row else None
            }
            report['anomalous_days'].append(anomaly_data)
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"JSON report saved as {filename}")
        return report
    
    def generate_html_report(self, filename='anomaly_report.html'):
        """Generate and save an HTML report."""
        if self.data is None:
            raise ValueError("No data available for report generation")
        
        # Prepare data for HTML table
        report_data = self.data.copy()
        report_data['Date'] = report_data.index.strftime('%Y-%m-%d')
        report_data['Daily_Return_Pct'] = (report_data['Daily_Return'] * 100).round(2)
        report_data['Z_Score'] = report_data['Z_Score'].round(3)
        report_data['Close'] = report_data['Close'].round(2)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.ticker} Stock Anomaly Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
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
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .anomaly {{
            background-color: #ffebee !important;
            font-weight: bold;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.ticker} Stock Anomaly Detection Report</h1>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p><strong>Analysis Period:</strong> {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}</p>
            <p><strong>Trading Days Analyzed:</strong> {len(self.data)}</p>
            <p><strong>Z-Score Threshold:</strong> ±{self.z_threshold}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(self.anomalies)}</div>
                <div class="stat-label">Anomalous Days</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{(self.data['Daily_Return'].mean() * 100):.2f}%</div>
                <div class="stat-label">Mean Daily Return</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{(self.data['Daily_Return'].std() * 100):.2f}%</div>
                <div class="stat-label">Return Volatility</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${self.data['Close'].iloc[-1]:.2f}</div>
                <div class="stat-label">Latest Close Price</div>
            </div>
        </div>
        
        <h2>Daily Trading Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Anomaly Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add table rows
        for _, row in report_data.iterrows():
            anomaly_class = "anomaly" if row['Is_Anomaly'] else ""
            return_class = "positive" if row['Daily_Return'] > 0 else "negative"
            anomaly_status = "🚨 ANOMALY" if row['Is_Anomaly'] else "Normal"
            
            html_content += f"""
                <tr class="{anomaly_class}">
                    <td>{row['Date']}</td>
                    <td>${row['Close']}</td>
                    <td class="{return_class}">{row['Daily_Return_Pct']:.2f}%</td>
                    <td>{row['Z_Score']:.3f}</td>
                    <td>{anomaly_status}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="summary">
            <h2>Methodology</h2>
            <p>This analysis identifies anomalous trading days using Z-score analysis of daily returns:</p>
            <ul>
                <li><strong>Daily Return:</strong> Calculated as (Close_today - Close_yesterday) / Close_yesterday</li>
                <li><strong>Z-Score:</strong> (Daily_Return - Mean_Return) / Standard_Deviation_Return</li>
                <li><strong>Anomaly Threshold:</strong> |Z-Score| > 2.0 (approximately 95% confidence interval)</li>
            </ul>
            <p>Days with Z-scores beyond ±2.0 are considered anomalous, representing unusually large price movements relative to the recent trading pattern.</p>
        </div>
        
        <div class="summary">
            <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved as {filename}")
    
    def run_analysis(self):
        """Run the complete anomaly detection analysis."""
        print(f"Starting anomaly detection analysis for {self.ticker}...")
        print("=" * 60)
        
        # Step 1: Fetch data
        if not self.fetch_stock_data():
            return False
        
        # Step 2: Calculate anomalies
        try:
            self.calculate_anomalies()
        except Exception as e:
            print(f"Error calculating anomalies: {str(e)}")
            return False
        
        # Step 3: Generate reports
        try:
            self.generate_chart()
            self.generate_json_report()
            self.generate_html_report()
        except Exception as e:
            print(f"Error generating reports: {str(e)}")
            return False
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("Generated files:")
        print("  - anomalies.png (chart)")
        print("  - anomaly_report.json (JSON report)")
        print("  - anomaly_report.html (HTML report)")
        
        return True


def main():
    """Main entry point for the stock anomaly detection script."""
    try:
        # Initialize detector with default parameters
        detector = StockAnomalyDetector(ticker='META', days=14, z_threshold=2.0)
        
        # Run the complete analysis
        success = detector.run_analysis()
        
        if success:
            print("\nAnalysis Summary:")
            if detector.anomalies is not None and len(detector.anomalies) > 0:
                print(f"Found {len(detector.anomalies)} anomalous trading days:")
                for date, row in detector.anomalies.iterrows():
                    return_pct = row['Daily_Return'] * 100
                    print(f"  - {date.strftime('%Y-%m-%d')}: {return_pct:+.2f}% (Z-score: {row['Z_Score']:.2f})")
            else:
                print("No anomalous trading days detected in the analyzed period.")
        else:
            print("Analysis failed. Please check the error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
