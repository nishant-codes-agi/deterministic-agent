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
    """Detects anomalies in stock trading data using Z-score analysis."""
    
    def __init__(self, ticker: str, days: int = 14, z_threshold: float = 2.0):
        self.ticker = ticker
        self.days = days
        self.z_threshold = z_threshold
        self.data: Optional[pd.DataFrame] = None
        self.anomalies: List[Dict] = []
        
    def fetch_stock_data(self) -> pd.DataFrame:
        """Fetch stock data from yfinance for the specified period."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Extra days to ensure we get enough trading days
            
            print(f"Fetching {self.ticker} stock data from {start_date.date()} to {end_date.date()}...")
            
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {self.ticker}")
            
            # Keep only the last 14 trading days
            data = data.tail(self.days)
            
            if len(data) < 2:
                raise ValueError(f"Insufficient data: only {len(data)} trading days available")
            
            print(f"Successfully fetched {len(data)} trading days of data")
            self.data = data
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock data for {self.ticker}: {str(e)}")
    
    def calculate_daily_returns(self) -> pd.Series:
        """Calculate daily returns using percentage change of adjusted close price."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data() first.")
        
        daily_returns = self.data['Adj Close'].pct_change().dropna()
        
        if daily_returns.empty:
            raise ValueError("Unable to calculate daily returns")
        
        return daily_returns
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect anomalous trading days using Z-score analysis."""
        daily_returns = self.calculate_daily_returns()
        
        # Calculate Z-scores
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            print("Warning: Standard deviation is 0, no anomalies can be detected")
            return []
        
        z_scores = (daily_returns - mean_return) / std_return
        
        # Identify anomalies
        anomalous_mask = (np.abs(z_scores) > self.z_threshold)
        anomalous_dates = z_scores[anomalous_mask]
        
        self.anomalies = []
        for date, z_score in anomalous_dates.items():
            daily_return = daily_returns[date]
            self.anomalies.append({
                'date': date.strftime('%Y-%m-%d'),
                'daily_return': float(daily_return),
                'daily_return_pct': float(daily_return * 100),
                'z_score': float(z_score),
                'adj_close': float(self.data.loc[date, 'Adj Close']),
                'volume': int(self.data.loc[date, 'Volume'])
            })
        
        # Sort by absolute Z-score (most anomalous first)
        self.anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return self.anomalies
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data() first.")
        
        daily_returns = self.calculate_daily_returns()
        
        # Calculate statistics
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        min_return = daily_returns.min()
        max_return = daily_returns.max()
        
        start_date = self.data.index[0].strftime('%Y-%m-%d')
        end_date = self.data.index[-1].strftime('%Y-%m-%d')
        
        report = f"""# META Stock Anomaly Detection Report

## Analysis Period
- **Ticker Symbol**: {self.ticker}
- **Date Range**: {start_date} to {end_date}
- **Total Trading Days**: {len(self.data)}
- **Z-Score Threshold**: ±{self.z_threshold}

## Daily Returns Statistics
- **Mean Daily Return**: {mean_return:.4f} ({mean_return*100:.2f}%)
- **Standard Deviation**: {std_return:.4f} ({std_return*100:.2f}%)
- **Minimum Daily Return**: {min_return:.4f} ({min_return*100:.2f}%)
- **Maximum Daily Return**: {max_return:.4f} ({max_return*100:.2f}%)

## Anomaly Detection Results
"""
        
        if self.anomalies:
            report += f"**{len(self.anomalies)} anomalous trading day(s) detected:**\n\n"
            
            for i, anomaly in enumerate(self.anomalies, 1):
                direction = "📈 Positive" if anomaly['z_score'] > 0 else "📉 Negative"
                report += f"""### {i}. {anomaly['date']} - {direction} Anomaly
- **Daily Return**: {anomaly['daily_return_pct']:.2f}%
- **Z-Score**: {anomaly['z_score']:.2f}
- **Adjusted Close**: ${anomaly['adj_close']:.2f}
- **Volume**: {anomaly['volume']:,}

"""
        else:
            report += "**No anomalous trading days detected** in the analyzed period.\n\n"
            report += f"All daily returns had Z-scores within the ±{self.z_threshold} threshold.\n\n"
        
        report += f"""## Methodology
- **Anomaly Detection**: Z-score analysis with threshold of ±{self.z_threshold}
- **Daily Returns**: Calculated as percentage change in adjusted close price
- **Z-Score Formula**: (daily_return - mean_return) / standard_deviation
- **Data Source**: Yahoo Finance via yfinance library

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_json_report(self, filename: str = 'anomaly_report.json') -> None:
        """Save anomaly detection results to JSON file."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data() first.")
        
        daily_returns = self.calculate_daily_returns()
        
        report_data = {
            'metadata': {
                'ticker': self.ticker,
                'analysis_period': {
                    'start_date': self.data.index[0].strftime('%Y-%m-%d'),
                    'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
                    'total_trading_days': len(self.data)
                },
                'z_score_threshold': self.z_threshold,
                'generated_at': datetime.now().isoformat()
            },
            'statistics': {
                'mean_daily_return': float(daily_returns.mean()),
                'std_daily_return': float(daily_returns.std()),
                'min_daily_return': float(daily_returns.min()),
                'max_daily_return': float(daily_returns.max())
            },
            'anomalies': self.anomalies,
            'anomaly_count': len(self.anomalies)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"JSON report saved to {filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to save JSON report: {str(e)}")
    
    def create_visualization(self, filename: str = 'anomalies.png') -> None:
        """Create and save a matplotlib chart highlighting anomalous days."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_stock_data() first.")
        
        daily_returns = self.calculate_daily_returns()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{self.ticker} Stock Analysis - Last {self.days} Trading Days', fontsize=16, fontweight='bold')
        
        # Plot 1: Stock price with anomalous days highlighted
        ax1.plot(self.data.index, self.data['Adj Close'], 'b-', linewidth=2, label='Adjusted Close Price')
        
        # Highlight anomalous days on price chart
        for anomaly in self.anomalies:
            date = pd.to_datetime(anomaly['date'])
            price = anomaly['adj_close']
            color = 'red' if anomaly['z_score'] < 0 else 'green'
            ax1.scatter(date, price, color=color, s=100, zorder=5, alpha=0.8)
            ax1.annotate(f"Z={anomaly['z_score']:.1f}", 
                        xy=(date, price), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        ax1.set_title('Stock Price with Anomalous Days Highlighted')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Daily returns with Z-score threshold lines
        returns_pct = daily_returns * 100
        ax2.bar(daily_returns.index, returns_pct, alpha=0.7, color='lightblue', label='Daily Returns')
        
        # Highlight anomalous returns
        for anomaly in self.anomalies:
            date = pd.to_datetime(anomaly['date'])
            return_pct = anomaly['daily_return_pct']
            color = 'red' if anomaly['z_score'] < 0 else 'green'
            ax2.bar(date, return_pct, color=color, alpha=0.8)
        
        # Add Z-score threshold lines (approximate)
        mean_return_pct = daily_returns.mean() * 100
        std_return_pct = daily_returns.std() * 100
        upper_threshold = mean_return_pct + (self.z_threshold * std_return_pct)
        lower_threshold = mean_return_pct - (self.z_threshold * std_return_pct)
        
        ax2.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score +{self.z_threshold} threshold')
        ax2.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, label=f'Z-score -{self.z_threshold} threshold')
        ax2.axhline(y=mean_return_pct, color='black', linestyle='-', alpha=0.5, label='Mean return')
        
        ax2.set_title('Daily Returns with Anomaly Thresholds')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
            plt.close()
        except Exception as e:
            raise RuntimeError(f"Failed to save visualization: {str(e)}")


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("META Stock Anomaly Detection Analysis")
        print("=" * 60)
        
        # Initialize detector
        detector = StockAnomalyDetector(ticker='META', days=14, z_threshold=2.0)
        
        # Fetch data
        detector.fetch_stock_data()
        
        # Detect anomalies
        print("\nAnalyzing daily returns for anomalies...")
        anomalies = detector.detect_anomalies()
        
        # Generate and print markdown report
        print("\nGenerating markdown report...")
        markdown_report = detector.generate_markdown_report()
        print("\n" + "=" * 60)
        print("MARKDOWN REPORT")
        print("=" * 60)
        print(markdown_report)
        
        # Save JSON report
        print("\nSaving JSON report...")
        detector.save_json_report('anomaly_report.json')
        
        # Create and save visualization
        print("\nCreating visualization...")
        detector.create_visualization('anomalies.png')
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Files generated:")
        print(f"  - anomaly_report.json (JSON report)")
        print(f"  - anomalies.png (visualization)")
        print(f"\nSummary: {len(anomalies)} anomalous trading day(s) detected")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
