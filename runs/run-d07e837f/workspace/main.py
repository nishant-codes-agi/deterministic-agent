#!/usr/bin/env python3
"""
Stock Anomaly Detection Tool

Detects anomalous trading days for AAPL stock using z-score analysis on volume data.
Pulls data from yfinance for the last 90 days and reports top 5 anomalies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from typing import Tuple, Dict, Any


class StockAnomalyDetector:
    """Detects anomalous trading days using z-score analysis on volume data."""
    
    def __init__(self, ticker: str = "AAPL", days: int = 90, initial_threshold: float = 3.0):
        self.ticker = ticker
        self.days = days
        self.initial_threshold = initial_threshold
        self.adaptive_threshold = initial_threshold
        self.detection_method = "standard"
        
    def fetch_stock_data(self) -> pd.DataFrame:
        """Fetch stock data from yfinance for the specified period."""
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            
            # Fetch data using yfinance
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {self.ticker}")
                
            if len(data) < 10:
                raise ValueError(f"Insufficient data points ({len(data)}) for analysis")
                
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock data: {str(e)}")
    
    def calculate_volume_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-score for volume data."""
        try:
            # Extract volume data
            volumes = data['Volume'].copy()
            
            # Remove any zero or negative volumes
            volumes = volumes[volumes > 0]
            
            if len(volumes) < 5:
                raise ValueError("Insufficient valid volume data for z-score calculation")
            
            # Calculate mean and standard deviation
            volume_mean = volumes.mean()
            volume_std = volumes.std()
            
            if volume_std == 0:
                raise ValueError("Volume standard deviation is zero - cannot calculate z-scores")
            
            # Calculate z-scores
            data['Volume_ZScore'] = (data['Volume'] - volume_mean) / volume_std
            data['Volume_ZScore_Abs'] = np.abs(data['Volume_ZScore'])
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate z-scores: {str(e)}")
    
    def detect_anomalies(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect anomalous trading days using adaptive threshold strategy."""
        try:
            # First attempt with initial threshold
            anomalies = data[data['Volume_ZScore_Abs'] > self.initial_threshold].copy()
            
            detection_info = {
                'initial_threshold': self.initial_threshold,
                'final_threshold': self.initial_threshold,
                'method': 'standard',
                'total_anomalies': len(anomalies)
            }
            
            # If no anomalies found, use adaptive threshold strategy
            if len(anomalies) == 0:
                self.detection_method = "adaptive"
                # Try progressively lower thresholds
                for threshold in [2.5, 2.0, 1.5]:
                    anomalies = data[data['Volume_ZScore_Abs'] > threshold].copy()
                    if len(anomalies) > 0:
                        self.adaptive_threshold = threshold
                        detection_info.update({
                            'final_threshold': threshold,
                            'method': 'adaptive',
                            'total_anomalies': len(anomalies)
                        })
                        break
                
                # If still no anomalies, take top 5 by z-score
                if len(anomalies) == 0:
                    self.detection_method = "top_outliers"
                    anomalies = data.nlargest(5, 'Volume_ZScore_Abs').copy()
                    min_zscore = anomalies['Volume_ZScore_Abs'].min()
                    detection_info.update({
                        'final_threshold': min_zscore,
                        'method': 'top_outliers',
                        'total_anomalies': len(anomalies)
                    })
            
            # Sort by absolute z-score (descending) and take top 5
            anomalies = anomalies.sort_values('Volume_ZScore_Abs', ascending=False).head(5)
            
            return anomalies, detection_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect anomalies: {str(e)}")
    
    def generate_report(self, anomalies: pd.DataFrame, detection_info: Dict[str, Any], 
                       original_data: pd.DataFrame) -> str:
        """Generate a formatted summary report."""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"STOCK ANOMALY DETECTION REPORT - {self.ticker}")
            report_lines.append("=" * 80)
            report_lines.append(f"Analysis Period: Last {self.days} days")
            report_lines.append(f"Total Trading Days Analyzed: {len(original_data)}")
            report_lines.append(f"Detection Method: {detection_info['method']}")
            report_lines.append(f"Z-Score Threshold Used: {detection_info['final_threshold']:.2f}")
            
            if detection_info['method'] == 'adaptive':
                report_lines.append(f"(Adapted from initial threshold: {detection_info['initial_threshold']:.2f})")
            elif detection_info['method'] == 'top_outliers':
                report_lines.append("(No anomalies found with standard thresholds - showing top outliers)")
            
            report_lines.append(f"Total Anomalies Found: {detection_info['total_anomalies']}")
            report_lines.append("")
            
            if len(anomalies) > 0:
                report_lines.append("TOP 5 ANOMALOUS TRADING DAYS:")
                report_lines.append("-" * 80)
                report_lines.append(f"{'Date':<12} {'Volume':<15} {'Z-Score':<10} {'Severity':<10}")
                report_lines.append("-" * 80)
                
                for idx, (date, row) in enumerate(anomalies.iterrows(), 1):
                    date_str = date.strftime('%Y-%m-%d')
                    volume = int(row['Volume'])
                    zscore = row['Volume_ZScore']
                    
                    # Determine severity
                    abs_zscore = abs(zscore)
                    if abs_zscore >= 3.0:
                        severity = "Extreme"
                    elif abs_zscore >= 2.0:
                        severity = "High"
                    elif abs_zscore >= 1.5:
                        severity = "Moderate"
                    else:
                        severity = "Low"
                    
                    report_lines.append(f"{date_str:<12} {volume:<15,} {zscore:<10.2f} {severity:<10}")
                
                # Add summary statistics
                report_lines.append("")
                report_lines.append("VOLUME STATISTICS:")
                report_lines.append("-" * 40)
                volume_stats = original_data['Volume'].describe()
                report_lines.append(f"Mean Volume: {volume_stats['mean']:,.0f}")
                report_lines.append(f"Std Deviation: {volume_stats['std']:,.0f}")
                report_lines.append(f"Min Volume: {volume_stats['min']:,.0f}")
                report_lines.append(f"Max Volume: {volume_stats['max']:,.0f}")
                
            else:
                report_lines.append("No anomalous trading days detected.")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate report: {str(e)}")
    
    def run_analysis(self) -> str:
        """Run the complete anomaly detection analysis."""
        try:
            # Fetch stock data
            print(f"Fetching {self.ticker} stock data for the last {self.days} days...", file=sys.stderr)
            data = self.fetch_stock_data()
            
            # Calculate z-scores
            print("Calculating volume z-scores...", file=sys.stderr)
            data_with_zscores = self.calculate_volume_zscore(data)
            
            # Detect anomalies
            print("Detecting anomalies...", file=sys.stderr)
            anomalies, detection_info = self.detect_anomalies(data_with_zscores)
            
            # Generate report
            print("Generating report...", file=sys.stderr)
            report = self.generate_report(anomalies, detection_info, data)
            
            return report
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)


def main():
    """Main entry point for the stock anomaly detection tool."""
    try:
        # Initialize detector with default parameters
        detector = StockAnomalyDetector(
            ticker="AAPL",
            days=90,
            initial_threshold=3.0
        )
        
        # Run analysis and print report
        report = detector.run_analysis()
        print(report)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nFatal error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
