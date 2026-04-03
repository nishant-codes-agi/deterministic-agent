#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

This script downloads stock data for AAPL and detects anomalies using the IQR method.
It generates a visualization and saves results to JSON.
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


class StockAnomalyDetector:
    """Detects anomalies in stock price data using IQR method."""
    
    def __init__(self, ticker: str = "AAPL", period: str = "1y", iqr_multiplier: float = 1.5):
        self.ticker = ticker
        self.period = period
        self.iqr_multiplier = iqr_multiplier
        self.data: Optional[pd.DataFrame] = None
        self.anomalies: Optional[pd.DataFrame] = None
        
    def fetch_stock_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Fetching stock data for {self.ticker} for period {self.period}...")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
                
            print(f"Successfully fetched {len(self.data)} data points")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Clean and preprocess the stock data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None:
                print("Error: No data to preprocess")
                return False
            
            # Check if Close column exists
            if 'Close' not in self.data.columns:
                print("Error: 'Close' column not found in data")
                return False
            
            # Handle missing values
            initial_count = len(self.data)
            self.data = self.data.dropna(subset=['Close'])
            final_count = len(self.data)
            
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} rows with missing Close prices")
            
            if self.data.empty:
                print("Error: No valid data remaining after preprocessing")
                return False
                
            print(f"Data preprocessing complete. {final_count} valid data points")
            return True
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return False
    
    def detect_anomalies(self) -> bool:
        """Detect anomalies using IQR method.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for anomaly detection")
                return False
            
            close_prices = self.data['Close']
            
            # Calculate quartiles and IQR
            q1 = close_prices.quantile(0.25)
            q3 = close_prices.quantile(0.75)
            iqr = q3 - q1
            
            # Define anomaly bounds
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            print(f"Anomaly detection parameters:")
            print(f"  Q1: ${q1:.2f}")
            print(f"  Q3: ${q3:.2f}")
            print(f"  IQR: ${iqr:.2f}")
            print(f"  Lower bound: ${lower_bound:.2f}")
            print(f"  Upper bound: ${upper_bound:.2f}")
            
            # Identify anomalies
            anomaly_mask = (close_prices < lower_bound) | (close_prices > upper_bound)
            self.anomalies = self.data[anomaly_mask].copy()
            
            print(f"Detected {len(self.anomalies)} anomalies")
            return True
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return False
    
    def print_anomalies(self) -> None:
        """Print detected anomalies to console."""
        if self.anomalies is None or self.anomalies.empty:
            print("No anomalies detected")
            return
        
        print("\nDetected Anomalies:")
        print("-" * 50)
        for date, row in self.anomalies.iterrows():
            print(f"Date: {date.strftime('%Y-%m-%d')}, Close Price: ${row['Close']:.2f}")
    
    def save_visualization(self, filename: str = "anomalies.png") -> bool:
        """Save visualization of stock data with anomalies highlighted.
        
        Args:
            filename: Output filename for the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for visualization")
                return False
            
            plt.figure(figsize=(12, 8))
            
            # Plot normal data points
            plt.plot(self.data.index, self.data['Close'], 
                    color='blue', alpha=0.7, linewidth=1, label='Close Price')
            
            # Highlight anomalies
            if self.anomalies is not None and not self.anomalies.empty:
                plt.scatter(self.anomalies.index, self.anomalies['Close'], 
                           color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)
            
            plt.title(f'{self.ticker} Stock Price with Anomaly Detection (IQR Method)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Close Price ($)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False
    
    def save_report(self, filename: str = "anomaly_report.json") -> bool:
        """Save anomaly detection report to JSON file.
        
        Args:
            filename: Output filename for the JSON report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None:
                print("Error: No data available for report generation")
                return False
            
            # Calculate daily returns
            returns = self.data['Close'].pct_change().dropna()
            
            # Prepare anomaly data
            anomaly_data = []
            if self.anomalies is not None and not self.anomalies.empty:
                for date, row in self.anomalies.iterrows():
                    # Calculate return for this date if possible
                    try:
                        daily_return = returns.loc[date] if date in returns.index else None
                    except:
                        daily_return = None
                    
                    anomaly_data.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "close_price": float(row['Close']),
                        "daily_return": float(daily_return) if daily_return is not None else None,
                        "volume": float(row.get('Volume', 0))
                    })
            
            # Create comprehensive report
            report = {
                "metadata": {
                    "ticker": self.ticker,
                    "period": self.period,
                    "iqr_multiplier": self.iqr_multiplier,
                    "detection_method": "IQR",
                    "generated_at": datetime.now().isoformat(),
                    "total_data_points": len(self.data),
                    "anomalies_detected": len(anomaly_data)
                },
                "statistics": {
                    "close_price_stats": {
                        "mean": float(self.data['Close'].mean()),
                        "std": float(self.data['Close'].std()),
                        "min": float(self.data['Close'].min()),
                        "max": float(self.data['Close'].max()),
                        "q1": float(self.data['Close'].quantile(0.25)),
                        "median": float(self.data['Close'].median()),
                        "q3": float(self.data['Close'].quantile(0.75))
                    },
                    "returns_stats": {
                        "mean": float(returns.mean()),
                        "std": float(returns.std()),
                        "min": float(returns.min()),
                        "max": float(returns.max())
                    }
                },
                "anomalies": anomaly_data
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Report saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def run_analysis(self) -> bool:
        """Run complete anomaly detection analysis.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Starting anomaly detection analysis for {self.ticker}")
        print("=" * 60)
        
        # Step 1: Fetch data
        if not self.fetch_stock_data():
            return False
        
        # Step 2: Preprocess data
        if not self.preprocess_data():
            return False
        
        # Step 3: Detect anomalies
        if not self.detect_anomalies():
            return False
        
        # Step 4: Print results
        self.print_anomalies()
        
        # Step 5: Save visualization
        if not self.save_visualization():
            print("Warning: Failed to save visualization")
        
        # Step 6: Save report
        if not self.save_report():
            print("Warning: Failed to save report")
        
        print("\nAnalysis completed successfully!")
        return True


def main():
    """Main entry point for the stock anomaly detection tool."""
    try:
        # Initialize detector with default parameters
        detector = StockAnomalyDetector(
            ticker="AAPL",
            period="1y",
            iqr_multiplier=1.5
        )
        
        # Run analysis
        success = detector.run_analysis()
        
        if not success:
            print("Analysis failed. Please check the error messages above.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
