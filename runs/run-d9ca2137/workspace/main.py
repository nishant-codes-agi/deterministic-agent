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
from typing import Tuple, List, Dict
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
        
        try:
            # Calculate daily percentage change
            daily_changes = self.data['Close'].pct_change() * 100
            
            # Remove NaN values (first day has no previous day to compare)
            daily_changes = daily_changes.dropna()
            
            print(f"Calculated daily changes for {len(daily_changes)} trading days")
            print(f"Average daily change: {daily_changes.mean():.2f}%")
            print(f"Standard deviation: {daily_changes.std():.2f}%")
            
            return daily_changes
            
        except Exception as e:
            print(f"Error calculating daily changes: {str(e)}")
            raise
    
    def detect_anomalies_iqr(self, daily_changes: pd.Series, multiplier: float = 1.5) -> Tuple[pd.Series, Dict]:
        """Detect anomalies using IQR method."""
        try:
            # Calculate quartiles and IQR
            Q1 = daily_changes.quantile(0.25)
            Q3 = daily_changes.quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify anomalies
            anomalies = daily_changes[(daily_changes < lower_bound) | (daily_changes > upper_bound)]
            
            stats = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(daily_changes) * 100
            }
            
            print(f"\nAnomaly Detection Results:")
            print(f"Q1 (25th percentile): {Q1:.2f}%")
            print(f"Q3 (75th percentile): {Q3:.2f}%")
            print(f"IQR: {IQR:.2f}%")
            print(f"Lower bound: {lower_bound:.2f}%")
            print(f"Upper bound: {upper_bound:.2f}%")
            print(f"Total anomalies detected: {len(anomalies)}")
            print(f"Anomaly rate: {stats['anomaly_rate']:.2f}%")
            
            self.anomalies = anomalies
            return anomalies, stats
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            raise
    
    def get_top_anomalies(self, anomalies: pd.Series, top_n: int = 5) -> List[Dict]:
        """Get top N anomalies sorted by absolute magnitude."""
        try:
            # Sort by absolute value of percentage change
            sorted_anomalies = anomalies.reindex(anomalies.abs().sort_values(ascending=False).index)
            
            # Get top N
            top_anomalies = sorted_anomalies.head(top_n)
            
            # Format results
            results = []
            for date, change in top_anomalies.items():
                # Get additional data for that day
                day_data = self.data.loc[date]
                
                # Determine if it's a rise or drop
                direction = "significant rise" if change > 0 else "significant drop"
                
                result = {
                    'date': date.strftime('%Y-%m-%d'),
                    'day_of_week': date.strftime('%A'),
                    'percentage_change': change,
                    'direction': direction,
                    'open_price': day_data['Open'],
                    'close_price': day_data['Close'],
                    'volume': day_data['Volume']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error getting top anomalies: {str(e)}")
            raise
    
    def print_summary(self, top_anomalies: List[Dict]):
        """Print a formatted summary of top anomalies."""
        print(f"\n{'='*80}")
        print(f"TOP {len(top_anomalies)} ANOMALOUS TRADING DAYS FOR {self.ticker}")
        print(f"{'='*80}")
        
        for i, anomaly in enumerate(top_anomalies, 1):
            print(f"\n{i}. {anomaly['date']} ({anomaly['day_of_week']})")
            print(f"   Percentage Change: {anomaly['percentage_change']:+.2f}% ({anomaly['direction']})")
            print(f"   Open: ${anomaly['open_price']:.2f}")
            print(f"   Close: ${anomaly['close_price']:.2f}")
            print(f"   Volume: {anomaly['volume']:,} shares")
            print(f"   {'-'*50}")
    
    def run_analysis(self) -> Dict:
        """Run complete anomaly detection analysis."""
        try:
            # Fetch data
            self.fetch_data()
            
            # Calculate daily changes
            daily_changes = self.calculate_daily_changes()
            
            # Detect anomalies
            anomalies, stats = self.detect_anomalies_iqr(daily_changes)
            
            # Get top anomalies
            top_anomalies = self.get_top_anomalies(anomalies)
            
            # Print summary
            self.print_summary(top_anomalies)
            
            return {
                'stats': stats,
                'top_anomalies': top_anomalies,
                'total_trading_days': len(daily_changes)
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    try:
        print("AAPL Stock Anomaly Detection Analysis")
        print("=" * 50)
        
        # Initialize detector
        detector = StockAnomalyDetector(ticker="AAPL", years=2)
        
        # Run analysis
        results = detector.run_analysis()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"Total trading days analyzed: {results['total_trading_days']}")
        print(f"Total anomalies detected: {results['stats']['total_anomalies']}")
        print(f"Anomaly rate: {results['stats']['anomaly_rate']:.2f}%")
        print(f"{'='*80}")
        
        return results
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
