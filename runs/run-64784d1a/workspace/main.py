#!/usr/bin/env python3
"""
TSLA Stock Anomaly Detection

This script pulls TSLA stock data for the last 14 days, detects anomalous trading days
using Z-score analysis, and generates reports in multiple formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TSLAAnomalyDetector:
    """
    A class to detect anomalous trading days in TSLA stock data.
    """
    
    def __init__(self, symbol: str = "TSLA", days: int = 14, z_threshold: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to look back
            z_threshold: Z-score threshold for anomaly detection
        """
        self.symbol = symbol
        self.days = days
        self.z_threshold = z_threshold
        self.data = None
        self.anomalies = []
        
    def fetch_stock_data(self) -> bool:
        """
        Fetch stock data from yfinance.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching {self.symbol} data for last {self.days} days...")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days + 5)  # Add buffer for weekends
            
            # Fetch data
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if self.data.empty:
                logger.error("No data retrieved from yfinance")
                return False
                
            # Keep only the last 14 trading days
            self.data = self.data.tail(self.days)
            
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Remove first row with NaN return
            self.data = self.data.dropna()
            
            logger.info(f"Successfully fetched {len(self.data)} trading days")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return False
    
    def detect_anomalies(self) -> List[Dict]:
        """
        Detect anomalous trading days using Z-score analysis.
        
        Returns:
            List of dictionaries containing anomaly information
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for anomaly detection")
            return []
            
        try:
            # Calculate Z-scores for daily returns
            returns = self.data['Daily_Return']
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                logger.warning("Standard deviation is zero, no anomalies can be detected")
                return []
            
            self.data['Z_Score'] = (returns - mean_return) / std_return
            
            # Identify anomalies
            anomaly_mask = (abs(self.data['Z_Score']) > self.z_threshold)
            anomaly_data = self.data[anomaly_mask]
            
            self.anomalies = []
            for date, row in anomaly_data.iterrows():
                anomaly = {
                    'date': date.strftime('%Y-%m-%d'),
                    'close_price': round(row['Close'], 2),
                    'daily_return': round(row['Daily_Return'] * 100, 2),  # Convert to percentage
                    'z_score': round(row['Z_Score'], 2),
                    'volume': int(row['Volume']),
                    'anomaly_type': 'positive' if row['Z_Score'] > 0 else 'negative'
                }
                self.anomalies.append(anomaly)
            
            logger.info(f"Detected {len(self.anomalies)} anomalous trading days")
            return self.anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def create_visualization(self, filename: str = "anomalies.png") -> bool:
        """
        Create and save a visualization of the stock data with anomalies highlighted.
        
        Args:
            filename: Output filename for the chart
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for visualization")
            return False
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Stock price with anomalous days highlighted
            dates = self.data.index
            prices = self.data['Close']
            
            ax1.plot(dates, prices, 'b-', linewidth=2, label='TSLA Close Price')
            
            # Highlight anomalous days
            for anomaly in self.anomalies:
                anomaly_date = pd.to_datetime(anomaly['date'])
                if anomaly_date in self.data.index:
                    price = self.data.loc[anomaly_date, 'Close']
                    color = 'red' if anomaly['anomaly_type'] == 'negative' else 'green'
                    ax1.scatter(anomaly_date, price, color=color, s=100, zorder=5,
                              label=f"Anomaly ({anomaly['anomaly_type']})" if anomaly == self.anomalies[0] else "")
            
            ax1.set_title(f'{self.symbol} Stock Price - Last {self.days} Trading Days', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Daily returns with Z-scores
            returns_pct = self.data['Daily_Return'] * 100
            z_scores = self.data['Z_Score']
            
            ax2.bar(dates, returns_pct, alpha=0.7, color='lightblue', label='Daily Returns (%)')
            
            # Highlight anomalous returns
            for anomaly in self.anomalies:
                anomaly_date = pd.to_datetime(anomaly['date'])
                if anomaly_date in self.data.index:
                    return_val = anomaly['daily_return']
                    color = 'red' if anomaly['anomaly_type'] == 'negative' else 'green'
                    ax2.bar(anomaly_date, return_val, color=color, alpha=0.8)
            
            # Add Z-score threshold lines
            ax2.axhline(y=self.z_threshold * self.data['Daily_Return'].std() * 100, 
                       color='red', linestyle='--', alpha=0.7, label=f'Z-score ±{self.z_threshold} threshold')
            ax2.axhline(y=-self.z_threshold * self.data['Daily_Return'].std() * 100, 
                       color='red', linestyle='--', alpha=0.7)
            
            ax2.set_title('Daily Returns with Anomaly Detection', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Daily Return (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return False
    
    def generate_json_report(self, filename: str = "anomaly_report.json") -> bool:
        """
        Generate and save a JSON report of the analysis.
        
        Args:
            filename: Output filename for the JSON report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate summary statistics
            if self.data is not None and not self.data.empty:
                summary_stats = {
                    'mean_return': round(self.data['Daily_Return'].mean() * 100, 2),
                    'std_return': round(self.data['Daily_Return'].std() * 100, 2),
                    'min_return': round(self.data['Daily_Return'].min() * 100, 2),
                    'max_return': round(self.data['Daily_Return'].max() * 100, 2),
                    'total_trading_days': len(self.data)
                }
            else:
                summary_stats = {}
            
            report = {
                'analysis_metadata': {
                    'symbol': self.symbol,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'period_days': self.days,
                    'z_threshold': self.z_threshold,
                    'data_source': 'yfinance'
                },
                'summary_statistics': summary_stats,
                'anomalies_detected': len(self.anomalies),
                'anomalous_days': self.anomalies
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"JSON report saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return False
    
    def generate_markdown_report(self, filename: str = "anomaly_report.md") -> bool:
        """
        Generate and save a markdown summary report.
        
        Args:
            filename: Output filename for the markdown report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            report_lines = []
            report_lines.append(f"# {self.symbol} Stock Anomaly Detection Report")
            report_lines.append("")
            report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Period:** Last {self.days} trading days")
            report_lines.append(f"**Z-Score Threshold:** ±{self.z_threshold}")
            report_lines.append(f"**Data Source:** yfinance")
            report_lines.append("")
            
            # Summary Statistics
            if self.data is not None and not self.data.empty:
                report_lines.append("## Summary Statistics")
                report_lines.append("")
                report_lines.append(f"- **Total Trading Days Analyzed:** {len(self.data)}")
                report_lines.append(f"- **Mean Daily Return:** {self.data['Daily_Return'].mean() * 100:.2f}%")
                report_lines.append(f"- **Standard Deviation:** {self.data['Daily_Return'].std() * 100:.2f}%")
                report_lines.append(f"- **Minimum Daily Return:** {self.data['Daily_Return'].min() * 100:.2f}%")
                report_lines.append(f"- **Maximum Daily Return:** {self.data['Daily_Return'].max() * 100:.2f}%")
                report_lines.append("")
            
            # Anomaly Detection Results
            report_lines.append("## Anomaly Detection Results")
            report_lines.append("")
            report_lines.append(f"**Total Anomalies Detected:** {len(self.anomalies)}")
            report_lines.append("")
            
            if self.anomalies:
                report_lines.append("### Anomalous Trading Days")
                report_lines.append("")
                report_lines.append("| Date | Close Price | Daily Return | Z-Score | Volume | Type |")
                report_lines.append("|------|-------------|--------------|---------|--------|------|")
                
                for anomaly in self.anomalies:
                    report_lines.append(
                        f"| {anomaly['date']} | ${anomaly['close_price']} | "
                        f"{anomaly['daily_return']:+.2f}% | {anomaly['z_score']:+.2f} | "
                        f"{anomaly['volume']:,} | {anomaly['anomaly_type'].title()} |"
                    )
                
                report_lines.append("")
                
                # Analysis by type
                positive_anomalies = [a for a in self.anomalies if a['anomaly_type'] == 'positive']
                negative_anomalies = [a for a in self.anomalies if a['anomaly_type'] == 'negative']
                
                report_lines.append("### Anomaly Breakdown")
                report_lines.append("")
                report_lines.append(f"- **Positive Anomalies (Unusual Gains):** {len(positive_anomalies)}")
                report_lines.append(f"- **Negative Anomalies (Unusual Losses):** {len(negative_anomalies)}")
                report_lines.append("")
                
                if positive_anomalies:
                    max_gain = max(positive_anomalies, key=lambda x: x['daily_return'])
                    report_lines.append(f"- **Largest Unusual Gain:** {max_gain['daily_return']:+.2f}% on {max_gain['date']}")
                
                if negative_anomalies:
                    max_loss = min(negative_anomalies, key=lambda x: x['daily_return'])
                    report_lines.append(f"- **Largest Unusual Loss:** {max_loss['daily_return']:+.2f}% on {max_loss['date']}")
                
            else:
                report_lines.append("No anomalous trading days detected in the analyzed period.")
                report_lines.append("")
                report_lines.append("This suggests that all daily returns were within the normal range ")
                report_lines.append(f"(Z-score between -{self.z_threshold} and +{self.z_threshold}).")
            
            report_lines.append("")
            report_lines.append("## Methodology")
            report_lines.append("")
            report_lines.append("This analysis uses Z-score normalization to identify anomalous trading days:")
            report_lines.append("")
            report_lines.append("1. **Data Collection:** Historical stock data retrieved from Yahoo Finance")
            report_lines.append("2. **Return Calculation:** Daily returns computed as (Close_t - Close_t-1) / Close_t-1")
            report_lines.append("3. **Z-Score Calculation:** Z = (Return - Mean) / Standard Deviation")
            report_lines.append(f"4. **Anomaly Detection:** Days with |Z-Score| > {self.z_threshold} flagged as anomalous")
            report_lines.append("")
            report_lines.append("## Files Generated")
            report_lines.append("")
            report_lines.append("- `anomalies.png` - Visualization of stock price and returns with anomalies highlighted")
            report_lines.append("- `anomaly_report.json` - Detailed JSON report with all analysis data")
            report_lines.append("- `anomaly_report.md` - This markdown summary report")
            
            # Write to file
            with open(filename, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Markdown report saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            return False
    
    def run_analysis(self) -> bool:
        """
        Run the complete anomaly detection analysis.
        
        Returns:
            bool: True if analysis completed successfully, False otherwise
        """
        logger.info("Starting TSLA anomaly detection analysis...")
        
        # Step 1: Fetch data
        if not self.fetch_stock_data():
            logger.error("Failed to fetch stock data")
            return False
        
        # Step 2: Detect anomalies
        self.detect_anomalies()
        
        # Step 3: Generate outputs
        success = True
        
        if not self.create_visualization():
            logger.warning("Failed to create visualization")
            success = False
        
        if not self.generate_json_report():
            logger.warning("Failed to generate JSON report")
            success = False
        
        if not self.generate_markdown_report():
            logger.warning("Failed to generate markdown report")
            success = False
        
        if success:
            logger.info("Analysis completed successfully!")
            logger.info(f"Found {len(self.anomalies)} anomalous trading days")
        else:
            logger.warning("Analysis completed with some errors")
        
        return success

def main():
    """
    Main entry point for the TSLA anomaly detection script.
    """
    try:
        # Initialize detector with default parameters
        detector = TSLAAnomalyDetector(
            symbol="TSLA",
            days=14,
            z_threshold=2.0
        )
        
        # Run the analysis
        success = detector.run_analysis()
        
        if success:
            print("\n" + "="*60)
            print("TSLA ANOMALY DETECTION ANALYSIS COMPLETE")
            print("="*60)
            print(f"Analyzed: {detector.symbol} stock for last {detector.days} trading days")
            print(f"Anomalies detected: {len(detector.anomalies)}")
            print("\nGenerated files:")
            print("  - anomalies.png (visualization)")
            print("  - anomaly_report.json (detailed data)")
            print("  - anomaly_report.md (summary report)")
            
            if detector.anomalies:
                print("\nAnomalous days found:")
                for anomaly in detector.anomalies:
                    print(f"  - {anomaly['date']}: {anomaly['daily_return']:+.2f}% "
                          f"(Z-score: {anomaly['z_score']:+.2f})")
            else:
                print("\nNo anomalous trading days detected.")
            
            print("\nAnalysis complete!")
            return 0
        else:
            print("Analysis failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
