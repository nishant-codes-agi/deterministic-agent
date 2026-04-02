#!/usr/bin/env python3
"""
AAPL Stock Anomaly Detection System

Fetches 2 years of AAPL stock data and detects anomalous trading days
using Interquartile Range (IQR) method.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockAnomalyDetector:
    """
    Detects anomalous trading days in stock data using IQR method.
    """
    
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.data = None
        self.anomalies = None
        
    def fetch_stock_data(self, years: int = 2) -> bool:
        """
        Fetch stock data for the specified number of years.
        
        Args:
            years: Number of years of historical data to fetch
            
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            logger.info(f"Fetching {years} years of {self.symbol} data from {start_date.date()} to {end_date.date()}")
            
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if self.data.empty:
                logger.error(f"No data retrieved for {self.symbol}")
                return False
                
            logger.info(f"Successfully fetched {len(self.data)} trading days of data")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return False
    
    def calculate_daily_metrics(self) -> None:
        """
        Calculate daily trading metrics for anomaly detection.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No stock data available. Call fetch_stock_data() first.")
            
        # Calculate daily price change percentage
        self.data['Price_Change_Pct'] = ((self.data['Close'] - self.data['Open']) / self.data['Open']) * 100
        
        # Calculate volume change from rolling average
        self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Change_Pct'] = ((self.data['Volume'] - self.data['Volume_MA_20']) / self.data['Volume_MA_20']) * 100
        
        # Calculate volatility (high-low range as percentage of open)
        self.data['Volatility_Pct'] = ((self.data['High'] - self.data['Low']) / self.data['Open']) * 100
        
        logger.info("Daily metrics calculated successfully")
    
    def detect_anomalies_iqr(self, metric: str = 'Price_Change_Pct', multiplier: float = 1.5) -> List[Dict]:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            metric: The metric to analyze for anomalies
            multiplier: IQR multiplier for outlier detection (1.5 is standard)
            
        Returns:
            List of dictionaries containing anomaly information
        """
        if self.data is None or metric not in self.data.columns:
            raise ValueError(f"Metric '{metric}' not available in data")
            
        # Remove NaN values for calculation
        clean_data = self.data[metric].dropna()
        
        if len(clean_data) == 0:
            logger.warning(f"No valid data points for metric '{metric}'")
            return []
        
        # Calculate IQR
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        logger.info(f"IQR bounds for {metric}: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Find anomalies
        anomaly_mask = (self.data[metric] < lower_bound) | (self.data[metric] > upper_bound)
        anomaly_data = self.data[anomaly_mask].copy()
        
        # Create anomaly records
        anomalies = []
        for date, row in anomaly_data.iterrows():
            anomaly = {
                'date': date.strftime('%Y-%m-%d'),
                'metric': metric,
                'value': row[metric],
                'deviation_from_median': abs(row[metric] - clean_data.median()),
                'deviation_pct': abs(row[metric] - clean_data.median()) / abs(clean_data.median()) * 100 if clean_data.median() != 0 else 0,
                'close_price': row['Close'],
                'volume': row['Volume'],
                'is_upper_outlier': row[metric] > upper_bound
            }
            anomalies.append(anomaly)
        
        # Sort by deviation percentage (descending)
        anomalies.sort(key=lambda x: x['deviation_pct'], reverse=True)
        
        logger.info(f"Detected {len(anomalies)} anomalies using IQR method")
        return anomalies
    
    def get_top_anomalies(self, n: int = 5) -> List[Dict]:
        """
        Get the top N anomalies by deviation percentage.
        
        Args:
            n: Number of top anomalies to return
            
        Returns:
            List of top anomaly dictionaries
        """
        if self.anomalies is None:
            logger.warning("No anomalies detected yet. Run detect_anomalies_iqr() first.")
            return []
            
        return self.anomalies[:n]
    
    def print_anomaly_summary(self, top_n: int = 5) -> None:
        """
        Print a formatted summary of the top anomalies.
        
        Args:
            top_n: Number of top anomalies to display
        """
        top_anomalies = self.get_top_anomalies(top_n)
        
        if not top_anomalies:
            print("No anomalies detected.")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {len(top_anomalies)} ANOMALOUS TRADING DAYS FOR {self.symbol}")
        print(f"{'='*80}")
        
        for i, anomaly in enumerate(top_anomalies, 1):
            direction = "📈 SURGE" if anomaly['is_upper_outlier'] else "📉 DROP"
            print(f"\n{i}. {anomaly['date']} - {direction}")
            print(f"   Price Change: {anomaly['value']:.2f}%")
            print(f"   Deviation: {anomaly['deviation_pct']:.2f}% from median")
            print(f"   Close Price: ${anomaly['close_price']:.2f}")
            print(f"   Volume: {anomaly['volume']:,} shares")
        
        print(f"\n{'='*80}")
    
    def run_analysis(self, years: int = 2, top_n: int = 5) -> Dict:
        """
        Run complete anomaly analysis pipeline.
        
        Args:
            years: Number of years of data to analyze
            top_n: Number of top anomalies to return
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Fetch data
            if not self.fetch_stock_data(years):
                return {'error': 'Failed to fetch stock data'}
            
            # Calculate metrics
            self.calculate_daily_metrics()
            
            # Detect anomalies
            self.anomalies = self.detect_anomalies_iqr()
            
            # Get top anomalies
            top_anomalies = self.get_top_anomalies(top_n)
            
            # Print summary
            self.print_anomaly_summary(top_n)
            
            return {
                'symbol': self.symbol,
                'analysis_period_years': years,
                'total_trading_days': len(self.data),
                'total_anomalies_detected': len(self.anomalies),
                'top_anomalies': top_anomalies,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg, 'success': False}

def main():
    """
    Main execution function.
    """
    print("🔍 AAPL Stock Anomaly Detection System")
    print("Analyzing 2 years of trading data...\n")
    
    # Initialize detector
    detector = StockAnomalyDetector("AAPL")
    
    # Run analysis
    results = detector.run_analysis(years=2, top_n=5)
    
    if results.get('success'):
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Analyzed {results['total_trading_days']} trading days")
        print(f"🚨 Found {results['total_anomalies_detected']} anomalous days")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
