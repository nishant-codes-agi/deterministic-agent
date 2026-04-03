#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

This script fetches stock data for AAPL and detects anomalies using the IQR method.
It generates a visualization and saves results to JSON.
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance library not found. Please install it using: pip install yfinance")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas library not found. Please install it using: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy library not found. Please install it using: pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib library not found. Please install it using: pip install matplotlib")
    sys.exit(1)


class StockAnomalyDetector:
    """Detects anomalies in stock price data using IQR method."""
    
    def __init__(self, ticker: str = "AAPL", period: str = "1y"):
        self.ticker = ticker
        self.period = period
        self.data: Optional[pd.DataFrame] = None
        self.anomalies: Optional[pd.DataFrame] = None
        
    def fetch_data(self) -> bool:
        """Fetch stock data from yfinance.
        
        Returns:
            bool: True if data was successfully fetched, False otherwise.
        """
        try:
            print(f"Fetching {self.ticker} stock data for period: {self.period}")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                print(f"Error: No data returned for ticker {self.ticker}")
                return False
                
            # Check if Close column exists
            if 'Close' not in self.data.columns:
                print("Error: Close price column not found in data")
                return False
                
            # Handle missing data by dropping NaN values
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=['Close'])
            final_rows = len(self.data)
            
            if initial_rows != final_rows:
                print(f"Removed {initial_rows - final_rows} rows with missing Close prices")
                
            if self.data.empty:
                print("Error: No valid data remaining after cleaning")
                return False
                
            print(f"Successfully fetched {len(self.data)} data points")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def detect_anomalies(self) -> bool:
        """Detect anomalies using IQR method.
        
        Returns:
            bool: True if anomaly detection was successful, False otherwise.
        """
        if self.data is None or self.data.empty:
            print("Error: No data available for anomaly detection")
            return False
            
        try:
            close_prices = self.data['Close']
            
            # Calculate quartiles and IQR
            Q1 = close_prices.quantile(0.25)
            Q3 = close_prices.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for anomaly detection
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            print(f"Anomaly detection bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            
            # Identify anomalies
            anomaly_mask = (close_prices < lower_bound) | (close_prices > upper_bound)
            self.anomalies = self.data[anomaly_mask].copy()
            
            if not self.anomalies.empty:
                self.anomalies['anomaly_type'] = self.anomalies['Close'].apply(
                    lambda x: 'low' if x < lower_bound else 'high'
                )
                
            print(f"Detected {len(self.anomalies)} anomalies")
            return True
            
        except Exception as e:
            print(f"Error during anomaly detection: {str(e)}")
            return False
    
    def print_anomalies(self) -> None:
        """Print detected anomalies to console."""
        if self.anomalies is None or self.anomalies.empty:
            print("No anomalies detected.")
            return
            
        print("\n=== DETECTED ANOMALIES ===")
        for date, row in self.anomalies.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            close_price = row['Close']
            anomaly_type = row['anomaly_type']
            print(f"Date: {date_str}, Close Price: ${close_price:.2f}, Type: {anomaly_type}")
    
    def save_chart(self, filename: str = "anomalies.png") -> bool:
        """Save a matplotlib chart highlighting anomalous days.
        
        Args:
            filename: Name of the output file.
            
        Returns:
            bool: True if chart was saved successfully, False otherwise.
        """
        if self.data is None or self.data.empty:
            print("Error: No data available for chart generation")
            return False
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot normal data points
            plt.plot(self.data.index, self.data['Close'], 
                    color='blue', alpha=0.7, linewidth=1, label='Close Price')
            
            # Highlight anomalies
            if self.anomalies is not None and not self.anomalies.empty:
                high_anomalies = self.anomalies[self.anomalies['anomaly_type'] == 'high']
                low_anomalies = self.anomalies[self.anomalies['anomaly_type'] == 'low']
                
                if not high_anomalies.empty:
                    plt.scatter(high_anomalies.index, high_anomalies['Close'], 
                              color='red', s=50, alpha=0.8, label='High Anomalies', zorder=5)
                              
                if not low_anomalies.empty:
                    plt.scatter(low_anomalies.index, low_anomalies['Close'], 
                              color='orange', s=50, alpha=0.8, label='Low Anomalies', zorder=5)
            
            plt.title(f'{self.ticker} Stock Price with Anomaly Detection (IQR Method)', fontsize=14)
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
            
            print(f"Chart saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving chart: {str(e)}")
            return False
    
    def save_json_report(self, filename: str = "anomaly_report.json") -> bool:
        """Save anomaly report to JSON file.
        
        Args:
            filename: Name of the output JSON file.
            
        Returns:
            bool: True if report was saved successfully, False otherwise.
        """
        try:
            report = {
                "ticker": self.ticker,
                "period": self.period,
                "analysis_date": datetime.now().isoformat(),
                "total_data_points": len(self.data) if self.data is not None else 0,
                "anomalies_detected": len(self.anomalies) if self.anomalies is not None else 0,
                "anomalies": []
            }
            
            if self.anomalies is not None and not self.anomalies.empty:
                for date, row in self.anomalies.iterrows():
                    # Calculate daily return if possible
                    daily_return = None
                    if len(self.data) > 1:
                        prev_close = self.data['Close'].shift(1).loc[date]
                        if pd.notna(prev_close) and prev_close != 0:
                            daily_return = ((row['Close'] - prev_close) / prev_close) * 100
                    
                    anomaly_entry = {
                        "date": date.strftime('%Y-%m-%d'),
                        "close_price": round(float(row['Close']), 2),
                        "anomaly_type": row['anomaly_type'],
                        "daily_return_percent": round(float(daily_return), 2) if daily_return is not None else None
                    }
                    report["anomalies"].append(anomaly_entry)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"JSON report saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON report: {str(e)}")
            return False
    
    def run_analysis(self) -> bool:
        """Run complete anomaly detection analysis.
        
        Returns:
            bool: True if analysis completed successfully, False otherwise.
        """
        print(f"Starting anomaly detection analysis for {self.ticker}")
        
        # Fetch data
        if not self.fetch_data():
            return False
            
        # Detect anomalies
        if not self.detect_anomalies():
            return False
            
        # Print results
        self.print_anomalies()
        
        # Save outputs
        chart_success = self.save_chart()
        json_success = self.save_json_report()
        
        if chart_success and json_success:
            print("\nAnalysis completed successfully!")
            return True
        else:
            print("\nAnalysis completed with some errors in output generation.")
            return False


def main():
    """Main entry point for the stock anomaly detection tool."""
    try:
        # Initialize detector with AAPL stock for 1 year period
        detector = StockAnomalyDetector(ticker="AAPL", period="1y")
        
        # Run the complete analysis
        success = detector.run_analysis()
        
        if success:
            print("\nFiles generated:")
            print("- anomalies.png: Chart showing stock prices with anomalies highlighted")
            print("- anomaly_report.json: Detailed report of detected anomalies")
        else:
            print("\nAnalysis failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
