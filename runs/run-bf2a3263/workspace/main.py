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
from typing import Dict, List, Tuple, Optional


class StockAnomalyDetector:
    """Detects anomalies in stock trading data using Z-score analysis."""
    
    def __init__(self, ticker: str = 'META', days: int = 14, z_threshold: float = 2.0):
        self.ticker = ticker
        self.days = days
        self.z_threshold = z_threshold
        self.data: Optional[pd.DataFrame] = None
        self.anomalies: Optional[pd.DataFrame] = None
        
    def fetch_stock_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if data was successfully fetched, False otherwise
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.days + 5)  # Extra days for weekends
            
            print(f"Fetching {self.ticker} data from {start_date} to {end_date}...")
            
            # Fetch data with error handling
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date)
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
                
            # Keep only the last 14 trading days
            self.data = self.data.tail(self.days)
            
            if len(self.data) < 2:
                print(f"Error: Insufficient data. Only {len(self.data)} trading days found.")
                return False
                
            print(f"Successfully fetched {len(self.data)} trading days of data.")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_anomalies(self) -> bool:
        """Calculate daily returns and detect anomalies using Z-score.
        
        Returns:
            bool: True if calculation was successful, False otherwise
        """
        if self.data is None or self.data.empty:
            print("Error: No data available for anomaly calculation.")
            return False
            
        try:
            # Calculate daily returns using adjusted close price
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove NaN values for statistical calculations
            returns_clean = self.data['Daily_Return'].dropna()
            
            if len(returns_clean) < 2:
                print("Error: Insufficient return data for Z-score calculation.")
                return False
            
            # Calculate mean and standard deviation
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            
            if std_return == 0:
                print("Warning: Standard deviation is zero. No anomalies can be detected.")
                return False
            
            # Calculate Z-scores
            self.data['Z_Score'] = (self.data['Daily_Return'] - mean_return) / std_return
            
            # Identify anomalies
            anomaly_mask = (abs(self.data['Z_Score']) > self.z_threshold) & (~self.data['Daily_Return'].isna())
            self.anomalies = self.data[anomaly_mask].copy()
            
            print(f"Calculated returns and Z-scores. Found {len(self.anomalies)} anomalous days.")
            return True
            
        except Exception as e:
            print(f"Error calculating anomalies: {str(e)}")
            return False
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report.
        
        Returns:
            str: Markdown formatted report
        """
        if self.data is None or self.anomalies is None:
            return "# Error\n\nNo data available for report generation."
        
        # Calculate statistics
        returns_clean = self.data['Daily_Return'].dropna()
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        # Get date range
        start_date = self.data.index[0].strftime('%Y-%m-%d')
        end_date = self.data.index[-1].strftime('%Y-%m-%d')
        
        # Build markdown report
        report = f"""# META Stock Anomaly Detection Report

## Analysis Period
- **Stock Ticker**: {self.ticker}
- **Date Range**: {start_date} to {end_date}
- **Total Trading Days Analyzed**: {len(self.data)}
- **Z-Score Threshold**: ±{self.z_threshold}

## Summary Statistics
- **Mean Daily Return**: {mean_return:.4f} ({mean_return*100:.2f}%)
- **Standard Deviation**: {std_return:.4f} ({std_return*100:.2f}%)
- **Anomalous Days Detected**: {len(self.anomalies)}

## Anomalous Trading Days
"""
        
        if len(self.anomalies) > 0:
            report += "\n| Date | Daily Return | Z-Score | Severity |\n"
            report += "|------|--------------|---------|----------|\n"
            
            for date, row in self.anomalies.iterrows():
                severity = "High" if abs(row['Z_Score']) > 3 else "Moderate"
                report += f"| {date.strftime('%Y-%m-%d')} | {row['Daily_Return']:.4f} ({row['Daily_Return']*100:.2f}%) | {row['Z_Score']:.2f} | {severity} |\n"
        else:
            report += "\nNo anomalous trading days detected during this period.\n"
        
        report += f"\n## Methodology\n\nThis analysis uses Z-score normalization to identify trading days with unusual returns:\n- Daily returns are calculated as percentage change in closing price\n- Z-scores are calculated using the mean and standard deviation of all returns in the period\n- Days with |Z-score| > {self.z_threshold} are flagged as anomalous\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def save_chart(self, filename: str = 'anomalies.png') -> bool:
        """Create and save a chart highlighting anomalous days.
        
        Args:
            filename: Output filename for the chart
            
        Returns:
            bool: True if chart was saved successfully, False otherwise
        """
        if self.data is None:
            print("Error: No data available for chart generation.")
            return False
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=1, label='Close Price')
            
            if len(self.anomalies) > 0:
                ax1.scatter(self.anomalies.index, 
                           [self.data.loc[date, 'Close'] for date in self.anomalies.index],
                           color='red', s=100, alpha=0.7, label='Anomalous Days', zorder=5)
            
            ax1.set_title(f'{self.ticker} Stock Price - Last {self.days} Trading Days')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Daily returns with Z-score threshold lines
            returns_data = self.data['Daily_Return'].dropna()
            ax2.bar(returns_data.index, returns_data * 100, alpha=0.6, color='blue', label='Daily Returns')
            
            if len(self.anomalies) > 0:
                anomaly_returns = self.anomalies['Daily_Return'] * 100
                ax2.bar(anomaly_returns.index, anomaly_returns, 
                       color='red', alpha=0.8, label='Anomalous Returns')
            
            # Add Z-score threshold lines (converted to percentage)
            mean_return = returns_data.mean()
            std_return = returns_data.std()
            upper_threshold = (mean_return + self.z_threshold * std_return) * 100
            lower_threshold = (mean_return - self.z_threshold * std_return) * 100
            
            ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score = +{self.z_threshold}')
            ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score = -{self.z_threshold}')
            
            ax2.set_title('Daily Returns with Anomaly Thresholds')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving chart: {str(e)}")
            return False
    
    def save_json_report(self, filename: str = 'anomaly_report.json') -> bool:
        """Save anomaly data to JSON file.
        
        Args:
            filename: Output filename for the JSON report
            
        Returns:
            bool: True if report was saved successfully, False otherwise
        """
        if self.data is None or self.anomalies is None:
            print("Error: No data available for JSON report generation.")
            return False
            
        try:
            # Prepare data for JSON serialization
            returns_clean = self.data['Daily_Return'].dropna()
            
            report_data = {
                'metadata': {
                    'ticker': self.ticker,
                    'analysis_period': {
                        'start_date': self.data.index[0].strftime('%Y-%m-%d'),
                        'end_date': self.data.index[-1].strftime('%Y-%m-%d')
                    },
                    'total_trading_days': len(self.data),
                    'z_score_threshold': self.z_threshold,
                    'generated_at': datetime.now().isoformat()
                },
                'statistics': {
                    'mean_daily_return': float(returns_clean.mean()),
                    'std_daily_return': float(returns_clean.std()),
                    'total_anomalies': len(self.anomalies)
                },
                'anomalous_days': []
            }
            
            # Add anomalous days data
            for date, row in self.anomalies.iterrows():
                anomaly_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'daily_return': float(row['Daily_Return']),
                    'daily_return_percent': float(row['Daily_Return'] * 100),
                    'z_score': float(row['Z_Score']),
                    'severity': 'High' if abs(row['Z_Score']) > 3 else 'Moderate',
                    'close_price': float(row['Close'])
                }
                report_data['anomalous_days'].append(anomaly_data)
            
            # Save to JSON file
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"JSON report saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON report: {str(e)}")
            return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("META Stock Anomaly Detection Analysis")
    print("=" * 60)
    
    # Initialize detector
    detector = StockAnomalyDetector(ticker='META', days=14, z_threshold=2.0)
    
    # Fetch data
    if not detector.fetch_stock_data():
        print("Failed to fetch stock data. Exiting.")
        return
    
    # Calculate anomalies
    if not detector.calculate_anomalies():
        print("Failed to calculate anomalies. Exiting.")
        return
    
    # Generate and display markdown report
    print("\n" + "=" * 60)
    print("MARKDOWN REPORT")
    print("=" * 60)
    markdown_report = detector.generate_markdown_report()
    print(markdown_report)
    
    # Save chart
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    chart_success = detector.save_chart('anomalies.png')
    json_success = detector.save_json_report('anomaly_report.json')
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Data fetched: ✓")
    print(f"Anomalies calculated: ✓")
    print(f"Chart saved: {'✓' if chart_success else '✗'}")
    print(f"JSON report saved: {'✓' if json_success else '✗'}")
    print(f"Anomalous days found: {len(detector.anomalies)}")
    
    if len(detector.anomalies) > 0:
        print("\nAnomalous dates:")
        for date in detector.anomalies.index:
            print(f"  - {date.strftime('%Y-%m-%d')}")
    else:
        print("\nNo anomalous trading days detected.")


if __name__ == "__main__":
    main()