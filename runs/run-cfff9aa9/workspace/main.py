#!/usr/bin/env python3
"""
META Stock Anomaly Detection

This script fetches META stock data for the last 14 days,
detects anomalous trading days using Z-score analysis,
and generates reports in multiple formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class StockAnomalyDetector:
    """Detects anomalous trading days in stock data using Z-score analysis."""
    
    def __init__(self, ticker: str, days: int = 14):
        self.ticker = ticker
        self.days = days
        self.data = None
        self.daily_returns = None
        self.anomalies = []
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from yfinance with robust error handling."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 10)  # Extra buffer for weekends
            
            print(f"Fetching {self.ticker} data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data with error handling
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                raise ValueError(f"No data returned for ticker {self.ticker}")
            
            # Check available columns and select price column
            available_columns = data.columns.tolist()
            print(f"Available columns: {available_columns}")
            
            # Priority order for price columns
            price_columns = ['Adj Close', 'Close', 'close', 'adj_close']
            selected_column = None
            
            for col in price_columns:
                if col in available_columns:
                    selected_column = col
                    break
            
            if selected_column is None:
                raise ValueError(f"No suitable price column found. Available: {available_columns}")
            
            print(f"Using price column: {selected_column}")
            
            # Keep only the last 14 trading days
            data = data.tail(self.days)
            
            if len(data) < 5:
                raise ValueError(f"Insufficient data: only {len(data)} trading days available")
            
            # Store the selected price column name for later use
            self.price_column = selected_column
            self.data = data
            
            print(f"Successfully fetched {len(data)} trading days of data")
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {self.ticker}: {str(e)}")
    
    def calculate_daily_returns(self) -> pd.Series:
        """Calculate daily returns using the available price column."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        try:
            # Calculate percentage change using the identified price column
            daily_returns = self.data[self.price_column].pct_change().dropna()
            
            if daily_returns.empty:
                raise ValueError("No valid daily returns calculated")
            
            self.daily_returns = daily_returns
            print(f"Calculated {len(daily_returns)} daily returns")
            return daily_returns
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate daily returns: {str(e)}")
    
    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        """Detect anomalous days using Z-score analysis."""
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        try:
            # Calculate Z-scores
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            
            if std_return == 0:
                print("Warning: Standard deviation is 0, no anomalies can be detected")
                return []
            
            z_scores = (self.daily_returns - mean_return) / std_return
            
            # Find anomalies
            anomalous_mask = (np.abs(z_scores) > threshold)
            anomalous_dates = z_scores[anomalous_mask]
            
            anomalies = []
            for date, z_score in anomalous_dates.items():
                daily_return = self.daily_returns[date]
                price = self.data.loc[date, self.price_column]
                
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'daily_return': float(daily_return),
                    'z_score': float(z_score),
                    'price': float(price),
                    'anomaly_type': 'positive' if z_score > 0 else 'negative'
                })
            
            self.anomalies = anomalies
            print(f"Detected {len(anomalies)} anomalous trading days")
            return anomalies
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect anomalies: {str(e)}")
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        if self.data is None or self.daily_returns is None:
            raise ValueError("No data available for report generation")
        
        try:
            # Calculate statistics
            start_date = self.data.index[0].strftime('%Y-%m-%d')
            end_date = self.data.index[-1].strftime('%Y-%m-%d')
            total_days = len(self.data)
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            
            # Build markdown report
            report = f"# {self.ticker} Stock Anomaly Detection Report\n\n"
            report += f"**Analysis Period:** {start_date} to {end_date}\n"
            report += f"**Total Trading Days:** {total_days}\n"
            report += f"**Mean Daily Return:** {mean_return:.4f} ({mean_return*100:.2f}%)\n"
            report += f"**Standard Deviation:** {std_return:.4f} ({std_return*100:.2f}%)\n\n"
            
            if self.anomalies:
                report += f"## Anomalous Trading Days ({len(self.anomalies)} detected)\n\n"
                report += "| Date | Daily Return | Z-Score | Price | Type |\n"
                report += "|------|--------------|---------|-------|------|\n"
                
                for anomaly in self.anomalies:
                    report += f"| {anomaly['date']} | {anomaly['daily_return']:.4f} ({anomaly['daily_return']*100:.2f}%) | {anomaly['z_score']:.2f} | ${anomaly['price']:.2f} | {anomaly['anomaly_type'].title()} |\n"
            else:
                report += "## No Anomalous Trading Days Detected\n\n"
                report += "All daily returns were within 2 standard deviations of the mean.\n"
            
            report += "\n---\n"
            report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
            
            return report
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate markdown report: {str(e)}")
    
    def save_chart(self, filename: str = 'anomalies.png') -> None:
        """Create and save a chart highlighting anomalous days."""
        if self.data is None or self.daily_returns is None:
            raise ValueError("No data available for chart generation")
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data[self.price_column], 'b-', linewidth=2, label='Stock Price')
            
            # Highlight anomalous days
            for anomaly in self.anomalies:
                date = pd.to_datetime(anomaly['date'])
                if date in self.data.index:
                    price = anomaly['price']
                    color = 'red' if anomaly['anomaly_type'] == 'negative' else 'green'
                    ax1.scatter(date, price, color=color, s=100, zorder=5, 
                              label=f"{anomaly['anomaly_type'].title()} Anomaly" if anomaly == self.anomalies[0] else "")
            
            ax1.set_title(f'{self.ticker} Stock Price - Last {self.days} Trading Days')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Daily returns with Z-score threshold lines
            ax2.plot(self.daily_returns.index, self.daily_returns * 100, 'b-', linewidth=2, label='Daily Returns')
            
            # Add threshold lines
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Calculate and plot Z-score thresholds in return space
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            upper_threshold = (mean_return + 2 * std_return) * 100
            lower_threshold = (mean_return - 2 * std_return) * 100
            
            ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label='Z-score = +2')
            ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, label='Z-score = -2')
            
            # Highlight anomalous returns
            for anomaly in self.anomalies:
                date = pd.to_datetime(anomaly['date'])
                if date in self.daily_returns.index:
                    return_pct = anomaly['daily_return'] * 100
                    color = 'red' if anomaly['anomaly_type'] == 'negative' else 'green'
                    ax2.scatter(date, return_pct, color=color, s=100, zorder=5)
            
            ax2.set_title('Daily Returns with Anomaly Detection')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved as {filename}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save chart: {str(e)}")
    
    def save_json_report(self, filename: str = 'anomaly_report.json') -> None:
        """Save anomaly detection results to JSON file."""
        if self.data is None or self.daily_returns is None:
            raise ValueError("No data available for JSON report generation")
        
        try:
            # Prepare comprehensive report data
            report_data = {
                'ticker': self.ticker,
                'analysis_period': {
                    'start_date': self.data.index[0].strftime('%Y-%m-%d'),
                    'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
                    'total_trading_days': len(self.data)
                },
                'statistics': {
                    'mean_daily_return': float(self.daily_returns.mean()),
                    'std_daily_return': float(self.daily_returns.std()),
                    'min_daily_return': float(self.daily_returns.min()),
                    'max_daily_return': float(self.daily_returns.max())
                },
                'anomalies': {
                    'count': len(self.anomalies),
                    'threshold_used': 2.0,
                    'detected_anomalies': self.anomalies
                },
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'price_column_used': self.price_column
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"JSON report saved as {filename}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save JSON report: {str(e)}")


def main():
    """Main execution function."""
    try:
        print("=== META Stock Anomaly Detection ===")
        print()
        
        # Initialize detector
        detector = StockAnomalyDetector('META', days=14)
        
        # Fetch data
        print("Step 1: Fetching stock data...")
        detector.fetch_data()
        print()
        
        # Detect anomalies
        print("Step 2: Detecting anomalies...")
        anomalies = detector.detect_anomalies()
        print()
        
        # Generate and print markdown report
        print("Step 3: Generating markdown report...")
        markdown_report = detector.generate_markdown_report()
        print(markdown_report)
        print()
        
        # Save chart
        print("Step 4: Saving chart...")
        detector.save_chart('anomalies.png')
        print()
        
        # Save JSON report
        print("Step 5: Saving JSON report...")
        detector.save_json_report('anomaly_report.json')
        print()
        
        print("=== Analysis Complete ===")
        print(f"Found {len(anomalies)} anomalous trading days")
        print("Files generated:")
        print("- anomalies.png (chart)")
        print("- anomaly_report.json (detailed report)")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()