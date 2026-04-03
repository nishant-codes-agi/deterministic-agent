#!/usr/bin/env python3
"""
Stock Data Analysis with Anomaly Detection

This script pulls stock data from Yahoo Finance using yfinance,
detects anomalies in daily returns, and generates visualizations and reports.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class StockAnomalyDetector:
    """
    A class to handle stock data retrieval and anomaly detection.
    """
    
    def __init__(self):
        self.data = None
        self.anomalies = None
        self.returns = None
    
    def get_stock_data(self, 
                      tickers: Union[str, List[str]], 
                      start_date: str, 
                      end_date: str) -> pd.DataFrame:
        """
        Retrieve stock data from Yahoo Finance.
        
        Args:
            tickers: Stock ticker symbol(s)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Stock data
            
        Raises:
            ValueError: For invalid input parameters
            Exception: For data retrieval errors
        """
        # Input validation
        if not tickers:
            raise ValueError("Tickers cannot be empty")
        
        # Convert single ticker to list for consistent handling
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Validate date formats
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
        
        # Data retrieval with error handling
        try:
            print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
            
            # Download data using yfinance
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                raise Exception(f"No data retrieved for tickers: {tickers}")
            
            # Handle MultiIndex columns for single ticker
            if len(tickers) == 1 and isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            print(f"Successfully retrieved {len(data)} rows of data")
            self.data = data
            return data
            
        except Exception as e:
            if "404" in str(e) or "Invalid ticker" in str(e):
                raise Exception(f"Invalid ticker symbol(s): {tickers}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise Exception(f"Network error while fetching data: {e}")
            else:
                raise Exception(f"Error retrieving stock data: {e}")
    
    def calculate_returns(self, price_column: str = 'Close') -> pd.Series:
        """
        Calculate daily returns from stock price data.
        
        Args:
            price_column: Column name to use for return calculation
            
        Returns:
            pandas.Series: Daily returns
        """
        if self.data is None:
            raise ValueError("No stock data available. Call get_stock_data first.")
        
        if price_column not in self.data.columns:
            available_cols = list(self.data.columns)
            raise ValueError(f"Column '{price_column}' not found. Available columns: {available_cols}")
        
        # Calculate daily returns (percentage change)
        returns = self.data[price_column].pct_change().dropna()
        self.returns = returns
        
        print(f"Calculated {len(returns)} daily returns")
        return returns
    
    def detect_anomalies_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            pandas.DataFrame: Anomalous data points
        """
        if self.returns is None:
            raise ValueError("No returns data available. Call calculate_returns first.")
        
        # Calculate Z-scores
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        z_scores = np.abs((self.returns - mean_return) / std_return)
        
        # Identify anomalies
        anomaly_mask = z_scores > threshold
        anomalous_dates = self.returns[anomaly_mask].index
        anomalous_returns = self.returns[anomaly_mask].values
        anomalous_zscores = z_scores[anomaly_mask].values
        
        # Create anomalies DataFrame
        anomalies_df = pd.DataFrame({
            'date': anomalous_dates,
            'return': anomalous_returns,
            'z_score': anomalous_zscores,
            'abs_return': np.abs(anomalous_returns)
        })
        
        # Sort by absolute return (most extreme first)
        anomalies_df = anomalies_df.sort_values('abs_return', ascending=False)
        
        self.anomalies = anomalies_df
        print(f"Detected {len(anomalies_df)} anomalies using Z-score method (threshold={threshold})")
        
        return anomalies_df
    
    def create_anomaly_chart(self, output_file: str = 'anomalies.png') -> None:
        """
        Create and save a matplotlib chart highlighting anomalous days.
        
        Args:
            output_file: Output filename for the chart
        """
        if self.returns is None or self.anomalies is None:
            raise ValueError("No data available for plotting. Run detection first.")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Stock price with anomaly markers
        ax1.plot(self.data.index, self.data['Close'], 'b-', linewidth=1, label='Close Price')
        
        # Mark anomalous dates on price chart
        for _, anomaly in self.anomalies.iterrows():
            date = anomaly['date']
            if date in self.data.index:
                price = self.data.loc[date, 'Close']
                color = 'red' if anomaly['return'] < 0 else 'green'
                ax1.scatter(date, price, color=color, s=100, alpha=0.7, 
                           marker='o', edgecolors='black', linewidth=1)
        
        ax1.set_title('Stock Price with Anomalous Days Highlighted', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Daily returns with anomaly markers
        ax2.plot(self.returns.index, self.returns * 100, 'gray', linewidth=0.8, alpha=0.7, label='Daily Returns')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark anomalous returns
        anomaly_returns = self.anomalies['return'] * 100
        anomaly_dates = self.anomalies['date']
        colors = ['red' if r < 0 else 'green' for r in anomaly_returns]
        
        ax2.scatter(anomaly_dates, anomaly_returns, c=colors, s=80, alpha=0.8, 
                   marker='o', edgecolors='black', linewidth=1, label='Anomalies')
        
        ax2.set_title('Daily Returns with Anomalies', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Daily Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add statistics text box
        stats_text = f"""Anomaly Statistics:
Total Anomalies: {len(self.anomalies)}
Mean Return: {self.returns.mean()*100:.2f}%
Std Return: {self.returns.std()*100:.2f}%
Max Anomaly: {self.anomalies['return'].max()*100:.2f}%
Min Anomaly: {self.anomalies['return'].min()*100:.2f}%"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Anomaly chart saved to {output_file}")
    
    def generate_anomaly_report(self, output_file: str = 'anomaly_report.json') -> Dict[str, Any]:
        """
        Generate and save a JSON report of anomalous dates and returns.
        
        Args:
            output_file: Output filename for the JSON report
            
        Returns:
            dict: The anomaly report
        """
        if self.returns is None or self.anomalies is None:
            raise ValueError("No data available for report. Run detection first.")
        
        # Prepare report data
        report = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'data_period': {
                    'start_date': self.data.index.min().strftime('%Y-%m-%d'),
                    'end_date': self.data.index.max().strftime('%Y-%m-%d'),
                    'total_trading_days': len(self.data)
                },
                'detection_method': 'Z-score',
                'threshold_used': 3.0
            },
            'summary_statistics': {
                'total_anomalies': len(self.anomalies),
                'anomaly_rate': len(self.anomalies) / len(self.returns),
                'mean_daily_return': float(self.returns.mean()),
                'std_daily_return': float(self.returns.std()),
                'max_positive_anomaly': float(self.anomalies['return'].max()) if len(self.anomalies) > 0 else None,
                'max_negative_anomaly': float(self.anomalies['return'].min()) if len(self.anomalies) > 0 else None
            },
            'anomalous_dates': []
        }
        
        # Add individual anomaly details
        for _, anomaly in self.anomalies.iterrows():
            anomaly_detail = {
                'date': anomaly['date'].strftime('%Y-%m-%d'),
                'daily_return': float(anomaly['return']),
                'daily_return_percent': float(anomaly['return'] * 100),
                'z_score': float(anomaly['z_score']),
                'severity': 'extreme' if anomaly['z_score'] > 4 else 'high' if anomaly['z_score'] > 3.5 else 'moderate'
            }
            report['anomalous_dates'].append(anomaly_detail)
        
        # Save report to JSON file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Anomaly report saved to {output_file}")
        return report


def main():
    """
    Main execution function.
    """
    # Initialize detector
    detector = StockAnomalyDetector()
    
    try:
        # Configuration
        ticker = "AAPL"  # Apple Inc.
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year of data
        
        print(f"Starting stock anomaly analysis for {ticker}")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 50)
        
        # Step 1: Get stock data
        stock_data = detector.get_stock_data(ticker, start_date, end_date)
        print(f"Data shape: {stock_data.shape}")
        print(f"Columns: {list(stock_data.columns)}")
        print()
        
        # Step 2: Calculate returns
        returns = detector.calculate_returns('Close')
        print(f"Returns statistics:")
        print(f"  Mean: {returns.mean()*100:.4f}%")
        print(f"  Std:  {returns.std()*100:.4f}%")
        print(f"  Min:  {returns.min()*100:.4f}%")
        print(f"  Max:  {returns.max()*100:.4f}%")
        print()
        
        # Step 3: Detect anomalies
        anomalies = detector.detect_anomalies_zscore(threshold=3.0)
        
        if len(anomalies) > 0:
            print("Top 5 most extreme anomalies:")
            print(anomalies.head().to_string(index=False))
            print()
        else:
            print("No anomalies detected with current threshold.")
            print()
        
        # Step 4: Create visualization
        detector.create_anomaly_chart('anomalies.png')
        
        # Step 5: Generate report
        report = detector.generate_anomaly_report('anomaly_report.json')
        
        print("-" * 50)
        print("Analysis completed successfully!")
        print(f"Files generated:")
        print(f"  - anomalies.png (visualization)")
        print(f"  - anomaly_report.json (detailed report)")
        
        # Display summary
        print(f"\nSummary:")
        print(f"  Total trading days analyzed: {len(stock_data)}")
        print(f"  Anomalies detected: {len(anomalies)}")
        print(f"  Anomaly rate: {len(anomalies)/len(returns)*100:.2f}%")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
