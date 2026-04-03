#!/usr/bin/env python3
"""
Stock Anomaly Detection System

This module pulls stock data from Yahoo Finance and detects anomalies
using statistical methods. It generates visualizations and reports.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StockAnomalyDetector:
    """
    A class to detect anomalies in stock price data using statistical methods.
    """
    
    def __init__(self, symbol: str = "AAPL", period: str = "1y"):
        """
        Initialize the anomaly detector.
        
        Args:
            symbol: Stock symbol to analyze (default: AAPL)
            period: Time period for data (default: 1y)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.anomalies = None
        
    def fetch_data(self) -> bool:
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            bool: True if data fetched successfully, False otherwise
        """
        try:
            print(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                print(f"No data found for symbol {self.symbol}")
                return False
                
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            self.data = self.data.dropna()
            
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def detect_anomalies_zscore(self, threshold: float = 2.5) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method on daily returns.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomalous data points
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        returns = self.data['Daily_Return']
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Calculate Z-scores
        z_scores = np.abs((returns - mean_return) / std_return)
        
        # Identify anomalies
        anomaly_mask = z_scores > threshold
        anomalies = self.data[anomaly_mask].copy()
        anomalies['Z_Score'] = z_scores[anomaly_mask]
        
        return anomalies
    
    def detect_anomalies_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            multiplier: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with anomalous data points
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        returns = self.data['Daily_Return']
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Identify anomalies
        anomaly_mask = (returns < lower_bound) | (returns > upper_bound)
        anomalies = self.data[anomaly_mask].copy()
        anomalies['Distance_from_Bounds'] = np.where(
            returns < lower_bound,
            lower_bound - returns,
            returns - upper_bound
        )[anomaly_mask]
        
        return anomalies
    
    def run_detection(self, method: str = "zscore", **kwargs) -> Dict:
        """
        Run anomaly detection using specified method.
        
        Args:
            method: Detection method ('zscore' or 'iqr')
            **kwargs: Additional parameters for the detection method
            
        Returns:
            Dictionary with detection results
        """
        if not self.fetch_data():
            return {"error": "Failed to fetch data"}
        
        try:
            if method == "zscore":
                threshold = kwargs.get('threshold', 2.5)
                self.anomalies = self.detect_anomalies_zscore(threshold)
                method_params = {"threshold": threshold}
            elif method == "iqr":
                multiplier = kwargs.get('multiplier', 1.5)
                self.anomalies = self.detect_anomalies_iqr(multiplier)
                method_params = {"multiplier": multiplier}
            else:
                return {"error": f"Unknown method: {method}"}
            
            print(f"Detected {len(self.anomalies)} anomalies using {method} method")
            
            return {
                "success": True,
                "method": method,
                "method_params": method_params,
                "anomaly_count": len(self.anomalies),
                "total_days": len(self.data),
                "anomaly_percentage": (len(self.anomalies) / len(self.data)) * 100
            }
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}
    
    def create_visualization(self, output_file: str = "anomalies.png") -> bool:
        """
        Create and save a visualization of the stock data with anomalies highlighted.
        
        Args:
            output_file: Output filename for the chart
            
        Returns:
            bool: True if visualization created successfully
        """
        if self.data is None or self.anomalies is None:
            print("No data or anomalies to visualize")
            return False
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Stock price with anomalous days highlighted
            ax1.plot(self.data.index, self.data['Close'], 
                    label=f'{self.symbol} Close Price', color='blue', alpha=0.7)
            
            if len(self.anomalies) > 0:
                ax1.scatter(self.anomalies.index, self.anomalies['Close'], 
                           color='red', s=50, alpha=0.8, 
                           label=f'Anomalous Days ({len(self.anomalies)})', zorder=5)
            
            ax1.set_title(f'{self.symbol} Stock Price with Anomaly Detection', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Daily returns with anomalies highlighted
            ax2.plot(self.data.index, self.data['Daily_Return'], 
                    label='Daily Returns', color='green', alpha=0.7)
            
            if len(self.anomalies) > 0:
                ax2.scatter(self.anomalies.index, self.anomalies['Daily_Return'], 
                           color='red', s=50, alpha=0.8, 
                           label=f'Anomalous Returns', zorder=5)
            
            ax2.set_title('Daily Returns with Anomalies', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Daily Return', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return False
    
    def generate_report(self, output_file: str = "anomaly_report.json") -> bool:
        """
        Generate and save a JSON report of anomalies.
        
        Args:
            output_file: Output filename for the report
            
        Returns:
            bool: True if report generated successfully
        """
        if self.data is None or self.anomalies is None:
            print("No data or anomalies to report")
            return False
        
        try:
            # Prepare anomaly data
            anomaly_list = []
            for date, row in self.anomalies.iterrows():
                anomaly_data = {
                    "date": date.strftime("%Y-%m-%d"),
                    "close_price": round(float(row['Close']), 2),
                    "daily_return": round(float(row['Daily_Return']), 6),
                    "volume": int(row['Volume'])
                }
                
                # Add method-specific metrics
                if 'Z_Score' in row:
                    anomaly_data["z_score"] = round(float(row['Z_Score']), 3)
                if 'Distance_from_Bounds' in row:
                    anomaly_data["distance_from_bounds"] = round(float(row['Distance_from_Bounds']), 6)
                
                anomaly_list.append(anomaly_data)
            
            # Calculate summary statistics
            returns = self.data['Daily_Return']
            summary_stats = {
                "mean_return": round(float(returns.mean()), 6),
                "std_return": round(float(returns.std()), 6),
                "min_return": round(float(returns.min()), 6),
                "max_return": round(float(returns.max()), 6),
                "total_trading_days": len(self.data)
            }
            
            # Create comprehensive report
            report = {
                "analysis_metadata": {
                    "symbol": self.symbol,
                    "period": self.period,
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_start_date": self.data.index[0].strftime("%Y-%m-%d"),
                    "data_end_date": self.data.index[-1].strftime("%Y-%m-%d")
                },
                "summary": {
                    "total_anomalies": len(self.anomalies),
                    "anomaly_percentage": round((len(self.anomalies) / len(self.data)) * 100, 2),
                    **summary_stats
                },
                "anomalies": anomaly_list
            }
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Report saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return False


def main():
    """
    Main function to run the stock anomaly detection system.
    """
    print("Stock Anomaly Detection System")
    print("=" * 40)
    
    # Configuration
    SYMBOL = "AAPL"  # Apple Inc.
    PERIOD = "1y"    # 1 year of data
    METHOD = "zscore" # Detection method
    
    # Initialize detector
    detector = StockAnomalyDetector(symbol=SYMBOL, period=PERIOD)
    
    # Run detection
    print(f"\nAnalyzing {SYMBOL} stock data...")
    result = detector.run_detection(method=METHOD, threshold=2.5)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Display results
    print(f"\nDetection Results:")
    print(f"- Method: {result['method']}")
    print(f"- Parameters: {result['method_params']}")
    print(f"- Total trading days: {result['total_days']}")
    print(f"- Anomalies detected: {result['anomaly_count']}")
    print(f"- Anomaly percentage: {result['anomaly_percentage']:.2f}%")
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    # Create visualization
    if detector.create_visualization("anomalies.png"):
        print("✓ Anomaly chart saved to anomalies.png")
    else:
        print("✗ Failed to create visualization")
    
    # Generate report
    if detector.generate_report("anomaly_report.json"):
        print("✓ Anomaly report saved to anomaly_report.json")
    else:
        print("✗ Failed to generate report")
    
    # Display some anomalous dates if found
    if len(detector.anomalies) > 0:
        print(f"\nTop 5 Most Extreme Anomalies:")
        if 'Z_Score' in detector.anomalies.columns:
            top_anomalies = detector.anomalies.nlargest(5, 'Z_Score')
            for date, row in top_anomalies.iterrows():
                print(f"- {date.strftime('%Y-%m-%d')}: Return = {row['Daily_Return']:.4f}, Z-Score = {row['Z_Score']:.2f}")
        else:
            top_anomalies = detector.anomalies.nlargest(5, 'Distance_from_Bounds')
            for date, row in top_anomalies.iterrows():
                print(f"- {date.strftime('%Y-%m-%d')}: Return = {row['Daily_Return']:.4f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()