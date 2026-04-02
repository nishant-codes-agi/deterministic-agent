#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

This script pulls AAPL stock data from yfinance for the last 90 days,
detects anomalous trading days using z-score on volume, and prints
a summary report with the top 5 anomaly dates.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from typing import Tuple, Optional


class StockAnomalyDetector:
    """Detects anomalous trading days based on volume z-scores."""
    
    def __init__(self, ticker: str = "AAPL", days: int = 90, z_threshold: float = 2.0):
        self.ticker = ticker
        self.days = days
        self.z_threshold = z_threshold
        self.data = None
        
    def fetch_stock_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if data was successfully fetched, False otherwise.
        """
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            
            print(f"Fetching {self.ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            
            # Download data
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if self.data.empty:
                print(f"Error: No data found for ticker {self.ticker}")
                return False
                
            if 'Volume' not in self.data.columns:
                print(f"Error: Volume data not available for ticker {self.ticker}")
                return False
                
            # Remove any rows with zero or NaN volume
            initial_rows = len(self.data)
            self.data = self.data[self.data['Volume'] > 0].dropna(subset=['Volume'])
            final_rows = len(self.data)
            
            if final_rows == 0:
                print("Error: No valid volume data found")
                return False
                
            if final_rows < initial_rows:
                print(f"Removed {initial_rows - final_rows} rows with invalid volume data")
                
            print(f"Successfully fetched {len(self.data)} trading days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def calculate_volume_zscore(self) -> pd.Series:
        """Calculate z-scores for volume data.
        
        Returns:
            pd.Series: Z-scores for each trading day.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Call fetch_stock_data() first.")
            
        volumes = self.data['Volume']
        mean_volume = volumes.mean()
        std_volume = volumes.std()
        
        if std_volume == 0:
            print("Warning: Standard deviation of volume is 0. All volumes are identical.")
            return pd.Series(0, index=volumes.index)
            
        z_scores = (volumes - mean_volume) / std_volume
        return z_scores
    
    def detect_anomalies(self) -> pd.DataFrame:
        """Detect anomalous trading days based on volume z-scores.
        
        Returns:
            pd.DataFrame: DataFrame with anomalous days, sorted by absolute z-score.
        """
        z_scores = self.calculate_volume_zscore()
        
        # Find anomalies where absolute z-score exceeds threshold
        anomaly_mask = np.abs(z_scores) > self.z_threshold
        anomalies = self.data[anomaly_mask].copy()
        
        if anomalies.empty:
            return pd.DataFrame()
            
        # Add z-scores to anomalies dataframe
        anomalies['Z_Score'] = z_scores[anomaly_mask]
        anomalies['Abs_Z_Score'] = np.abs(anomalies['Z_Score'])
        
        # Sort by absolute z-score in descending order
        anomalies = anomalies.sort_values('Abs_Z_Score', ascending=False)
        
        return anomalies
    
    def generate_report(self, top_n: int = 5) -> str:
        """Generate a summary report of anomalous trading days.
        
        Args:
            top_n (int): Number of top anomalies to include in report.
            
        Returns:
            str: Formatted report string.
        """
        if self.data is None:
            return "Error: No data available. Please fetch data first."
            
        anomalies = self.detect_anomalies()
        
        # Calculate basic statistics
        total_days = len(self.data)
        total_anomalies = len(anomalies)
        mean_volume = self.data['Volume'].mean()
        std_volume = self.data['Volume'].std()
        
        # Build report
        report_lines = [
            "="*80,
            f"STOCK ANOMALY DETECTION REPORT - {self.ticker}",
            "="*80,
            f"Analysis Period: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}",
            f"Total Trading Days: {total_days}",
            f"Z-Score Threshold: {self.z_threshold}",
            f"Total Anomalies Detected: {total_anomalies}",
            f"Anomaly Rate: {(total_anomalies/total_days)*100:.1f}%",
            "",
            f"Volume Statistics:",
            f"  Mean Volume: {mean_volume:,.0f}",
            f"  Std Deviation: {std_volume:,.0f}",
            "",
        ]
        
        if anomalies.empty:
            report_lines.extend([
                "No anomalous trading days detected with the current threshold.",
                "Consider lowering the z-score threshold to detect more subtle anomalies."
            ])
        else:
            top_anomalies = anomalies.head(top_n)
            report_lines.extend([
                f"TOP {min(top_n, len(anomalies))} ANOMALOUS TRADING DAYS:",
                "-"*80,
                f"{'Date':<12} {'Volume':<15} {'Z-Score':<10} {'Type':<12} {'Price Change':<12}",
                "-"*80
            ])
            
            for date, row in top_anomalies.iterrows():
                volume = row['Volume']
                z_score = row['Z_Score']
                anomaly_type = "High Volume" if z_score > 0 else "Low Volume"
                
                # Calculate price change
                price_change = ((row['Close'] - row['Open']) / row['Open']) * 100
                price_change_str = f"{price_change:+.2f}%"
                
                report_lines.append(
                    f"{date.strftime('%Y-%m-%d'):<12} {volume:<15,.0f} {z_score:<10.2f} {anomaly_type:<12} {price_change_str:<12}"
                )
            
            if len(anomalies) > top_n:
                report_lines.append(f"\n... and {len(anomalies) - top_n} more anomalies")
        
        report_lines.extend([
            "",
            "="*80,
            "Report generated successfully."
        ])
        
        return "\n".join(report_lines)


def main():
    """Main function to run the stock anomaly detection."""
    print("Stock Anomaly Detection Tool")
    print("Analyzing AAPL volume anomalies for the last 90 days...\n")
    
    try:
        # Initialize detector
        detector = StockAnomalyDetector(
            ticker="AAPL",
            days=90,
            z_threshold=2.0
        )
        
        # Fetch data
        if not detector.fetch_stock_data():
            print("Failed to fetch stock data. Exiting.")
            sys.exit(1)
        
        print("\nAnalyzing volume patterns...")
        
        # Generate and print report
        report = detector.generate_report(top_n=5)
        print("\n" + report)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
