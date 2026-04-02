#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

Fetches 2 weeks of AAPL stock data and detects anomalous trading days using IQR method.
Prints a summary of the top 5 anomalies with dates and percentage deviations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from typing import Tuple, Optional, List, Dict


class StockAnomalyDetector:
    """Detects anomalous trading days using IQR method."""
    
    def __init__(self, ticker: str = 'AAPL', days: int = 14, iqr_multiplier: float = 1.5):
        self.ticker = ticker
        self.days = days
        self.iqr_multiplier = iqr_multiplier
        self.data = None
        self.anomalies = None
    
    def fetch_stock_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            
            print(f"Fetching {self.days} days of {self.ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if self.data.empty:
                print(f"Error: No data retrieved for ticker {self.ticker}")
                return False
            
            if len(self.data) < 2:
                print(f"Error: Insufficient data points ({len(self.data)}). Need at least 2 days for percentage change calculation.")
                return False
            
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_percentage_changes(self) -> bool:
        """Calculate daily percentage changes in close prices.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for percentage change calculation")
                return False
            
            # Calculate percentage change
            self.data['pct_change'] = self.data['Close'].pct_change() * 100
            
            # Remove NaN values (first day will be NaN)
            self.data = self.data.dropna()
            
            if self.data.empty:
                print("Error: No valid percentage changes calculated")
                return False
            
            print(f"Calculated percentage changes for {len(self.data)} trading days")
            return True
            
        except Exception as e:
            print(f"Error calculating percentage changes: {str(e)}")
            return False
    
    def detect_anomalies(self) -> bool:
        """Detect anomalies using IQR method.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data is None or 'pct_change' not in self.data.columns:
                print("Error: No percentage change data available for anomaly detection")
                return False
            
            pct_changes = self.data['pct_change']
            
            # Calculate quartiles and IQR
            q1 = pct_changes.quantile(0.25)
            q3 = pct_changes.quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds for anomaly detection
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            print(f"IQR Analysis:")
            print(f"  Q1: {q1:.2f}%")
            print(f"  Q3: {q3:.2f}%")
            print(f"  IQR: {iqr:.2f}%")
            print(f"  Lower bound: {lower_bound:.2f}%")
            print(f"  Upper bound: {upper_bound:.2f}%")
            
            # Identify anomalies
            anomaly_mask = (pct_changes < lower_bound) | (pct_changes > upper_bound)
            self.anomalies = self.data[anomaly_mask].copy()
            
            if self.anomalies.empty:
                print("No anomalies detected using IQR method")
                return True
            
            # Calculate absolute deviation from the nearest bound
            self.anomalies['abs_deviation'] = self.anomalies['pct_change'].apply(
                lambda x: abs(x - upper_bound) if x > upper_bound else abs(x - lower_bound)
            )
            
            # Sort by absolute deviation (descending)
            self.anomalies = self.anomalies.sort_values('abs_deviation', ascending=False)
            
            print(f"Detected {len(self.anomalies)} anomalous trading days")
            return True
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return False
    
    def print_anomaly_summary(self, top_n: int = 5) -> None:
        """Print summary of top N anomalies.
        
        Args:
            top_n: Number of top anomalies to display
        """
        try:
            if self.anomalies is None:
                print("No anomaly data available")
                return
            
            if self.anomalies.empty:
                print("No anomalies detected")
                return
            
            print(f"\n{'='*60}")
            print(f"TOP {min(top_n, len(self.anomalies))} ANOMALOUS TRADING DAYS FOR {self.ticker}")
            print(f"{'='*60}")
            
            top_anomalies = self.anomalies.head(top_n)
            
            for i, (date, row) in enumerate(top_anomalies.iterrows(), 1):
                date_str = date.strftime('%Y-%m-%d')
                pct_change = row['pct_change']
                abs_deviation = row['abs_deviation']
                close_price = row['Close']
                
                print(f"{i}. Date: {date_str}")
                print(f"   Close Price: ${close_price:.2f}")
                print(f"   Percentage Change: {pct_change:+.2f}%")
                print(f"   Deviation from Normal: {abs_deviation:.2f}%")
                print()
            
        except Exception as e:
            print(f"Error printing anomaly summary: {str(e)}")
    
    def run_analysis(self) -> bool:
        """Run complete anomaly detection analysis.
        
        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        print(f"Starting anomaly detection analysis for {self.ticker}...\n")
        
        # Step 1: Fetch data
        if not self.fetch_stock_data():
            return False
        
        # Step 2: Calculate percentage changes
        if not self.calculate_percentage_changes():
            return False
        
        # Step 3: Detect anomalies
        if not self.detect_anomalies():
            return False
        
        # Step 4: Print summary
        self.print_anomaly_summary()
        
        return True


def main():
    """Main entry point for the stock anomaly detection tool."""
    try:
        # Initialize detector with default parameters
        detector = StockAnomalyDetector(
            ticker='AAPL',
            days=14,
            iqr_multiplier=1.5
        )
        
        # Run analysis
        success = detector.run_analysis()
        
        if not success:
            print("Analysis failed. Please check the error messages above.")
            sys.exit(1)
        
        print("Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
