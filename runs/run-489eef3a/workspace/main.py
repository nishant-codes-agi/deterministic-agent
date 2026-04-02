#!/usr/bin/env python3
"""
AAPL Stock Anomaly Detection

Fetches 2 years of daily AAPL stock data, detects anomalous trading days using IQR,
and prints a summary of the top 5 anomalies with their dates and percentage deviations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys


class StockAnomalyDetector:
    """Detects anomalous trading days using IQR method."""
    
    def __init__(self, ticker: str = "AAPL", years: int = 2):
        self.ticker = ticker
        self.years = years
        self.data = None
        self.anomalies = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from yfinance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.years * 365)
            
            print(f"Fetching {self.years} years of {self.ticker} data...")
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Download data with error handling
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            print(f"Successfully fetched {len(data)} trading days")
            self.data = data
            return data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_daily_changes(self) -> pd.Series:
        """Calculate daily percentage changes."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        # Calculate daily percentage change
        daily_changes = self.data['Close'].pct_change() * 100
        
        # Remove NaN values (first day has no previous day to compare)
        daily_changes = daily_changes.dropna()
        
        print(f"Calculated daily changes for {len(daily_changes)} trading days")
        return daily_changes
    
    def detect_anomalies_iqr(self, daily_changes: pd.Series, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """Detect anomalies using IQR method."""
        # Calculate quartiles and IQR
        Q1 = daily_changes.quantile(0.25)
        Q3 = daily_changes.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define anomaly thresholds
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        print(f"IQR Analysis:")
        print(f"  Q1 (25th percentile): {Q1:.2f}%")
        print(f"  Q3 (75th percentile): {Q3:.2f}%")
        print(f"  IQR: {IQR:.2f}%")
        print(f"  Lower bound: {lower_bound:.2f}%")
        print(f"  Upper bound: {upper_bound:.2f}%")
        
        # Identify anomalies
        anomaly_mask = (daily_changes < lower_bound) | (daily_changes > upper_bound)
        anomalies = daily_changes[anomaly_mask]
        
        # Create anomaly dataframe with additional information
        anomaly_df = pd.DataFrame({
            'Date': anomalies.index,
            'Percentage_Change': anomalies.values,
            'Absolute_Change': np.abs(anomalies.values),
            'Type': ['Significant Drop' if x < 0 else 'Significant Rise' for x in anomalies.values]
        })
        
        print(f"Found {len(anomaly_df)} anomalous trading days")
        self.anomalies = anomaly_df
        return anomaly_df
    
    def get_top_anomalies(self, n: int = 5) -> pd.DataFrame:
        """Get top N anomalies sorted by absolute magnitude."""
        if self.anomalies is None:
            raise ValueError("No anomalies detected. Call detect_anomalies_iqr() first.")
        
        # Sort by absolute change (magnitude) in descending order
        top_anomalies = self.anomalies.nlargest(n, 'Absolute_Change')
        return top_anomalies
    
    def print_summary(self, top_anomalies: pd.DataFrame):
        """Print formatted summary of top anomalies."""
        print("\n" + "="*80)
        print(f"TOP {len(top_anomalies)} ANOMALOUS TRADING DAYS FOR {self.ticker}")
        print("="*80)
        
        for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
            date_str = row['Date'].strftime('%Y-%m-%d (%A)')
            change = row['Percentage_Change']
            change_type = row['Type']
            
            print(f"\n{i}. {date_str}")
            print(f"   Percentage Change: {change:+.2f}%")
            print(f"   Classification: {change_type}")
            
            # Add context about magnitude
            if abs(change) > 10:
                print(f"   Magnitude: EXTREME (>{abs(change):.1f}%)")
            elif abs(change) > 5:
                print(f"   Magnitude: HIGH (>{abs(change):.1f}%)")
            else:
                print(f"   Magnitude: MODERATE ({abs(change):.1f}%)")
        
        print("\n" + "="*80)
    
    def run_analysis(self) -> Dict:
        """Run complete anomaly detection analysis."""
        try:
            # Fetch data
            self.fetch_data()
            
            # Calculate daily changes
            daily_changes = self.calculate_daily_changes()
            
            # Detect anomalies
            anomalies = self.detect_anomalies_iqr(daily_changes)
            
            # Get top 5 anomalies
            top_anomalies = self.get_top_anomalies(5)
            
            # Print summary
            self.print_summary(top_anomalies)
            
            # Return results for API use
            return {
                'ticker': self.ticker,
                'total_trading_days': len(daily_changes),
                'total_anomalies': len(anomalies),
                'top_anomalies': top_anomalies.to_dict('records')
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            raise


def create_api_endpoint():
    """Simple API endpoint function for external use."""
    def get_stock_anomalies(ticker: str = "AAPL", years: int = 2) -> Dict:
        """API endpoint to get stock anomalies."""
        detector = StockAnomalyDetector(ticker, years)
        return detector.run_analysis()
    
    return get_stock_anomalies


def main():
    """Main entry point."""
    print("AAPL Stock Anomaly Detection System")
    print("===================================\n")
    
    try:
        # Create detector instance
        detector = StockAnomalyDetector("AAPL", 2)
        
        # Run analysis
        results = detector.run_analysis()
        
        # Print additional statistics
        print(f"\nAnalysis Summary:")
        print(f"- Total trading days analyzed: {results['total_trading_days']}")
        print(f"- Total anomalies detected: {results['total_anomalies']}")
        print(f"- Anomaly rate: {(results['total_anomalies']/results['total_trading_days']*100):.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
