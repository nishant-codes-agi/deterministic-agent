#!/usr/bin/env python3
"""
META Stock Anomaly Detection

This script fetches META stock data for the last 14 days,
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
import warnings
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StockAnomalyDetector:
    """Class to handle stock data fetching and anomaly detection."""
    
    def __init__(self, ticker: str = 'META', days: int = 14):
        self.ticker = ticker
        self.days = days
        self.data = None
        self.anomalies = None
        
    def fetch_stock_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Extra days for weekends
            
            print(f"Fetching {self.ticker} stock data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data using yfinance
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            # Keep only the last 14 trading days
            self.data = self.data.tail(self.days)
            
            print(f"Successfully fetched {len(self.data)} trading days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_anomalies(self) -> bool:
        """Calculate daily returns and detect anomalies using Z-score.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                raise ValueError("No stock data available for analysis")
            
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove NaN values (first day has no previous day for comparison)
            returns = self.data['Daily_Return'].dropna()
            
            if len(returns) < 2:
                raise ValueError("Insufficient data for statistical analysis")
            
            # Calculate Z-scores
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                print("Warning: Standard deviation is zero, no anomalies can be detected")
                self.data['Z_Score'] = 0
                self.anomalies = pd.DataFrame()
                return True
            
            self.data['Z_Score'] = (self.data['Daily_Return'] - mean_return) / std_return
            
            # Identify anomalies (|Z-score| > 2)
            anomaly_mask = (abs(self.data['Z_Score']) > 2) & (~self.data['Daily_Return'].isna())
            self.anomalies = self.data[anomaly_mask].copy()
            
            print(f"Detected {len(self.anomalies)} anomalous trading days")
            return True
            
        except Exception as e:
            print(f"Error calculating anomalies: {str(e)}")
            return False
    
    def generate_chart(self, filename: str = 'anomalies.png') -> bool:
        """Generate and save a matplotlib chart highlighting anomalies.
        
        Args:
            filename: Output filename for the chart
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                raise ValueError("No data available for charting")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{self.ticker} Stock Analysis - Last {self.days} Trading Days', fontsize=16, fontweight='bold')
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=2, label='Close Price')
            
            if not self.anomalies.empty:
                ax1.scatter(self.anomalies.index, self.anomalies['Close'], 
                           color='red', s=100, zorder=5, label='Anomalous Days')
            
            ax1.set_title('Stock Price with Anomalous Days Highlighted')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            
            # Plot 2: Daily returns with Z-score threshold lines
            valid_data = self.data.dropna(subset=['Daily_Return'])
            ax2.bar(valid_data.index, valid_data['Daily_Return'] * 100, 
                   color=['red' if abs(z) > 2 else 'blue' for z in valid_data['Z_Score']], 
                   alpha=0.7)
            
            # Add threshold lines
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax2.set_title('Daily Returns (%) with Anomaly Threshold')
            ax2.set_ylabel('Daily Return (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            
            # Rotate x-axis labels for better readability
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            return False
    
    def generate_json_report(self, filename: str = 'anomaly_report.json') -> bool:
        """Generate and save a JSON report with anomaly details.
        
        Args:
            filename: Output filename for the JSON report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None:
                raise ValueError("No data available for report generation")
            
            # Prepare report data
            report = {
                'analysis_info': {
                    'ticker': self.ticker,
                    'analysis_period_days': self.days,
                    'start_date': self.data.index[0].strftime('%Y-%m-%d') if not self.data.empty else None,
                    'end_date': self.data.index[-1].strftime('%Y-%m-%d') if not self.data.empty else None,
                    'total_trading_days': len(self.data),
                    'anomalous_days_count': len(self.anomalies) if self.anomalies is not None else 0,
                    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'statistics': {},
                'anomalous_days': [],
                'all_trading_days': []
            }
            
            # Add statistics if we have valid data
            valid_returns = self.data['Daily_Return'].dropna()
            if not valid_returns.empty:
                report['statistics'] = {
                    'mean_daily_return': float(valid_returns.mean()),
                    'std_daily_return': float(valid_returns.std()),
                    'min_daily_return': float(valid_returns.min()),
                    'max_daily_return': float(valid_returns.max()),
                    'anomaly_threshold_zscore': 2.0
                }
            
            # Add anomalous days
            if self.anomalies is not None and not self.anomalies.empty:
                for date, row in self.anomalies.iterrows():
                    anomaly_data = {
                        'date': date.strftime('%Y-%m-%d'),
                        'close_price': float(row['Close']),
                        'daily_return': float(row['Daily_Return']) if not pd.isna(row['Daily_Return']) else None,
                        'daily_return_percent': float(row['Daily_Return'] * 100) if not pd.isna(row['Daily_Return']) else None,
                        'z_score': float(row['Z_Score']) if not pd.isna(row['Z_Score']) else None,
                        'anomaly_type': 'positive' if row['Z_Score'] > 2 else 'negative' if row['Z_Score'] < -2 else 'unknown'
                    }
                    report['anomalous_days'].append(anomaly_data)
            
            # Add all trading days for reference
            for date, row in self.data.iterrows():
                day_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'close_price': float(row['Close']),
                    'daily_return': float(row['Daily_Return']) if not pd.isna(row['Daily_Return']) else None,
                    'daily_return_percent': float(row['Daily_Return'] * 100) if not pd.isna(row['Daily_Return']) else None,
                    'z_score': float(row['Z_Score']) if not pd.isna(row['Z_Score']) else None,
                    'is_anomalous': bool(abs(row['Z_Score']) > 2) if not pd.isna(row['Z_Score']) else False
                }
                report['all_trading_days'].append(day_data)
            
            # Save to JSON file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"JSON report saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating JSON report: {str(e)}")
            return False
    
    def generate_html_report(self, filename: str = 'anomaly_report.html') -> bool:
        """Generate and save an HTML summary report.
        
        Args:
            filename: Output filename for the HTML report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None:
                raise ValueError("No data available for report generation")
            
            # Calculate summary statistics
            valid_returns = self.data['Daily_Return'].dropna()
            mean_return = valid_returns.mean() if not valid_returns.empty else 0
            std_return = valid_returns.std() if not valid_returns.empty else 0
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.ticker} Stock Anomaly Detection Report</title>
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
        .anomaly-highlight {{
            background-color: #ffebee;
            font-weight: bold;
        }}
        .positive-anomaly {{
            background-color: #e8f5e8;
            color: #2e7d32;
        }}
        .negative-anomaly {{
            background-color: #ffebee;
            color: #c62828;
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
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.ticker} Stock Anomaly Detection Report</h1>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{len(self.data)}</div>
                    <div>Trading Days Analyzed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(self.anomalies) if self.anomalies is not None else 0}</div>
                    <div>Anomalous Days Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{mean_return:.4f}</div>
                    <div>Mean Daily Return</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{std_return:.4f}</div>
                    <div>Return Std Deviation</div>
                </div>
            </div>
            <p><strong>Analysis Period:</strong> {self.data.index[0].strftime('%Y-%m-%d') if not self.data.empty else 'N/A'} to {self.data.index[-1].strftime('%Y-%m-%d') if not self.data.empty else 'N/A'}</p>
            <p><strong>Anomaly Threshold:</strong> Z-score > 2 or Z-score < -2</p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""
            
            # Add anomalies section if any exist
            if self.anomalies is not None and not self.anomalies.empty:
                html_content += """
        <h2>Detected Anomalies</h2>
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
                    anomaly_type = "Positive" if row['Z_Score'] > 2 else "Negative"
                    css_class = "positive-anomaly" if row['Z_Score'] > 2 else "negative-anomaly"
                    daily_return_pct = row['Daily_Return'] * 100 if not pd.isna(row['Daily_Return']) else 0
                    
                    html_content += f"""
                <tr class="{css_class}">
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>${row['Close']:.2f}</td>
                    <td>{daily_return_pct:.2f}%</td>
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
        <h2>Detected Anomalies</h2>
        <p style="text-align: center; color: #27ae60; font-size: 1.2em;">No anomalous trading days detected in the analysis period.</p>
"""
            
            # Add complete trading data table
            html_content += """
        <h2>Complete Trading Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Return (%)</th>
                    <th>Z-Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for date, row in self.data.iterrows():
                is_anomalous = not pd.isna(row['Z_Score']) and abs(row['Z_Score']) > 2
                css_class = ""
                status = "Normal"
                
                if is_anomalous:
                    if row['Z_Score'] > 2:
                        css_class = "positive-anomaly"
                        status = "Positive Anomaly"
                    else:
                        css_class = "negative-anomaly"
                        status = "Negative Anomaly"
                
                daily_return_pct = row['Daily_Return'] * 100 if not pd.isna(row['Daily_Return']) else 0
                z_score_display = f"{row['Z_Score']:.2f}" if not pd.isna(row['Z_Score']) else "N/A"
                
                html_content += f"""
                <tr class="{css_class}">
                    <td>{date.strftime('%Y-%m-%d')}</td>
                    <td>${row['Close']:.2f}</td>
                    <td>{daily_return_pct:.2f}%</td>
                    <td>{z_score_display}</td>
                    <td>{status}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
        
        <div style="margin-top: 30px; padding: 20px; background-color: #ecf0f1; border-radius: 5px;">
            <h3>Methodology</h3>
            <p><strong>Daily Return Calculation:</strong> (Close_today - Close_yesterday) / Close_yesterday</p>
            <p><strong>Z-Score Calculation:</strong> (Daily_Return - Mean_Return) / Standard_Deviation_Return</p>
            <p><strong>Anomaly Detection:</strong> Days with |Z-Score| > 2 are considered anomalous</p>
            <p><strong>Data Source:</strong> Yahoo Finance via yfinance library</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML report saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            return False

def main():
    """Main function to run the stock anomaly detection analysis."""
    print("=" * 60)
    print("META Stock Anomaly Detection Analysis")
    print("=" * 60)
    
    # Initialize detector
    detector = StockAnomalyDetector(ticker='META', days=14)
    
    # Step 1: Fetch stock data
    print("\n1. Fetching stock data...")
    if not detector.fetch_stock_data():
        print("Failed to fetch stock data. Exiting.")
        return False
    
    # Step 2: Calculate anomalies
    print("\n2. Calculating daily returns and detecting anomalies...")
    if not detector.calculate_anomalies():
        print("Failed to calculate anomalies. Exiting.")
        return False
    
    # Step 3: Generate chart
    print("\n3. Generating visualization chart...")
    if not detector.generate_chart('anomalies.png'):
        print("Failed to generate chart.")
    
    # Step 4: Generate JSON report
    print("\n4. Generating JSON report...")
    if not detector.generate_json_report('anomaly_report.json'):
        print("Failed to generate JSON report.")
    
    # Step 5: Generate HTML report
    print("\n5. Generating HTML report...")
    if not detector.generate_html_report('anomaly_report.html'):
        print("Failed to generate HTML report.")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if detector.data is not None and not detector.data.empty:
        print(f"Analyzed {len(detector.data)} trading days for {detector.ticker}")
        print(f"Period: {detector.data.index[0].strftime('%Y-%m-%d')} to {detector.data.index[-1].strftime('%Y-%m-%d')}")
        
        if detector.anomalies is not None:
            print(f"Detected {len(detector.anomalies)} anomalous trading days")
            
            if not detector.anomalies.empty:
                print("\nAnomalous days:")
                for date, row in detector.anomalies.iterrows():
                    return_pct = row['Daily_Return'] * 100 if not pd.isna(row['Daily_Return']) else 0
                    print(f"  {date.strftime('%Y-%m-%d')}: {return_pct:+.2f}% (Z-score: {row['Z_Score']:.2f})")
            else:
                print("No anomalous days detected.")
    
    print("\nGenerated files:")
    print("  - anomalies.png (visualization chart)")
    print("  - anomaly_report.json (detailed JSON report)")
    print("  - anomaly_report.html (summary HTML report)")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Please check the error messages above.")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your internet connection and try again.")
