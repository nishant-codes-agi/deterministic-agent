#!/usr/bin/env python3
"""
Stock Anomaly Detection System

Fetches 2 years of daily AAPL stock data, detects anomalous trading days using IQR,
and prints a summary of the top 5 anomalies with their dates and percentage deviations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from typing import Tuple, Optional


class StockAnomalyDetector:
    """Detects anomalous trading days using IQR method."""
    
    def __init__(self, ticker: str = "AAPL", years: int = 2):
        self.ticker = ticker
        self.years = years
        self.data = None
        self.anomalies = None
    
    def fetch_stock_data(self) -> bool:
        """
        Fetch historical stock data using yfinance.
        
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.years * 365)
            
            print(f"Fetching {self.years} years of {self.ticker} data from {start_date.date()} to {end_date.date()}...")
            
            # Download data
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date)
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
            
            print(f"Successfully fetched {len(self.data)} trading days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return False
    
    def calculate_daily_changes(self) -> bool:
        """
        Calculate daily percentage changes in closing price.
        
        Returns:
            bool: True if calculation successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for calculation")
                return False
            
            # Calculate daily percentage change
            self.data['Daily_Change_Pct'] = self.data['Close'].pct_change() * 100
            
            # Remove NaN values (first day will be NaN)
            self.data = self.data.dropna()
            
            print(f"Calculated daily percentage changes for {len(self.data)} trading days")
            return True
            
        except Exception as e:
            print(f"Error calculating daily changes: {e}")
            return False
    
    def detect_anomalies(self, iqr_multiplier: float = 1.5) -> bool:
        """
        Detect anomalies using IQR method.
        
        Args:
            iqr_multiplier: Multiplier for IQR to define outlier bounds
            
        Returns:
            bool: True if detection successful, False otherwise
        """
        try:
            if self.data is None or 'Daily_Change_Pct' not in self.data.columns:
                print("Error: Daily changes not calculated")
                return False
            
            # Calculate quartiles and IQR
            Q1 = self.data['Daily_Change_Pct'].quantile(0.25)
            Q3 = self.data['Daily_Change_Pct'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            print(f"IQR Analysis:")
            print(f"  Q1: {Q1:.2f}%")
            print(f"  Q3: {Q3:.2f}%")
            print(f"  IQR: {IQR:.2f}%")
            print(f"  Lower bound: {lower_bound:.2f}%")
            print(f"  Upper bound: {upper_bound:.2f}%")
            
            # Identify anomalies
            anomaly_mask = (
                (self.data['Daily_Change_Pct'] < lower_bound) |
                (self.data['Daily_Change_Pct'] > upper_bound)
            )
            
            self.anomalies = self.data[anomaly_mask].copy()
            
            if self.anomalies.empty:
                print("No anomalies detected")
                return True
            
            # Calculate absolute deviation for ranking
            self.anomalies['Abs_Deviation'] = abs(self.anomalies['Daily_Change_Pct'])
            
            # Sort by absolute deviation (descending)
            self.anomalies = self.anomalies.sort_values('Abs_Deviation', ascending=False)
            
            print(f"Detected {len(self.anomalies)} anomalous trading days")
            return True
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return False
    
    def print_top_anomalies(self, top_n: int = 5) -> None:
        """
        Print summary of top N anomalies.
        
        Args:
            top_n: Number of top anomalies to display
        """
        try:
            if self.anomalies is None or self.anomalies.empty:
                print("No anomalies to display")
                return
            
            top_anomalies = self.anomalies.head(top_n)
            
            print(f"\n{'='*60}")
            print(f"TOP {min(top_n, len(self.anomalies))} ANOMALOUS TRADING DAYS FOR {self.ticker}")
            print(f"{'='*60}")
            
            for i, (date, row) in enumerate(top_anomalies.iterrows(), 1):
                date_str = date.strftime('%Y-%m-%d')
                change_pct = row['Daily_Change_Pct']
                close_price = row['Close']
                volume = row['Volume']
                
                print(f"\n{i}. Date: {date_str}")
                print(f"   Daily Change: {change_pct:+.2f}%")
                print(f"   Closing Price: ${close_price:.2f}")
                print(f"   Volume: {volume:,}")
                
                # Add context about the magnitude
                if abs(change_pct) > 10:
                    magnitude = "EXTREME"
                elif abs(change_pct) > 5:
                    magnitude = "HIGH"
                else:
                    magnitude = "MODERATE"
                
                direction = "GAIN" if change_pct > 0 else "LOSS"
                print(f"   Magnitude: {magnitude} {direction}")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            print(f"Error printing anomalies: {e}")
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about the analysis.
        
        Returns:
            dict: Summary statistics
        """
        if self.data is None:
            return {}
        
        stats = {
            'total_trading_days': len(self.data),
            'anomalous_days': len(self.anomalies) if self.anomalies is not None else 0,
            'anomaly_rate': (len(self.anomalies) / len(self.data) * 100) if self.anomalies is not None else 0,
            'avg_daily_change': self.data['Daily_Change_Pct'].mean(),
            'std_daily_change': self.data['Daily_Change_Pct'].std(),
            'max_gain': self.data['Daily_Change_Pct'].max(),
            'max_loss': self.data['Daily_Change_Pct'].min()
        }
        
        return stats


def main():
    """
    Main function to run the stock anomaly detection analysis.
    """
    print("Stock Anomaly Detection System")
    print("=" * 40)
    
    # Initialize detector
    detector = StockAnomalyDetector(ticker="AAPL", years=2)
    
    # Step 1: Fetch data
    if not detector.fetch_stock_data():
        print("Failed to fetch stock data. Exiting.")
        sys.exit(1)
    
    # Step 2: Calculate daily changes
    if not detector.calculate_daily_changes():
        print("Failed to calculate daily changes. Exiting.")
        sys.exit(1)
    
    # Step 3: Detect anomalies
    if not detector.detect_anomalies():
        print("Failed to detect anomalies. Exiting.")
        sys.exit(1)
    
    # Step 4: Print results
    detector.print_top_anomalies(top_n=5)
    
    # Step 5: Print summary statistics
    stats = detector.get_summary_stats()
    if stats:
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total Trading Days: {stats['total_trading_days']}")
        print(f"Anomalous Days: {stats['anomalous_days']}")
        print(f"Anomaly Rate: {stats['anomaly_rate']:.2f}%")
        print(f"Average Daily Change: {stats['avg_daily_change']:+.2f}%")
        print(f"Standard Deviation: {stats['std_daily_change']:.2f}%")
        print(f"Maximum Gain: {stats['max_gain']:+.2f}%")
        print(f"Maximum Loss: {stats['max_loss']:+.2f}%")


if __name__ == "__main__":
    main()
