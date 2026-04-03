#!/usr/bin/env python3
"""
Stock Data Anomaly Detection System

This module fetches stock data from Yahoo Finance and detects anomalies
using statistical methods. It generates visualizations and reports for
anomalous trading days.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockAnomalyDetector:
    """
    A class to fetch stock data and detect anomalies in stock returns.
    """
    
    def __init__(self, z_threshold: float = 2.5):
        """
        Initialize the anomaly detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
        """
        self.z_threshold = z_threshold
        self.stock_data = {}
        self.anomalies = {}
    
    def get_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for given tickers and date range.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        logger.info(f"Fetching stock data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        stock_data = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}")
                
                # Create ticker object
                stock = yf.Ticker(ticker)
                
                # Fetch historical data
                data = stock.history(start=start_date, end=end_date)
                
                if data.empty:
                    logger.warning(f"No data found for ticker {ticker}")
                    stock_data[ticker] = pd.DataFrame()
                    continue
                
                # Calculate daily returns
                data['Daily_Return'] = data['Close'].pct_change()
                
                # Remove first row with NaN return
                data = data.dropna()
                
                if len(data) < 10:  # Minimum data points for meaningful analysis
                    logger.warning(f"Insufficient data for ticker {ticker} (only {len(data)} points)")
                    stock_data[ticker] = pd.DataFrame()
                    continue
                
                stock_data[ticker] = data
                logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                stock_data[ticker] = pd.DataFrame()
        
        self.stock_data = stock_data
        return stock_data
    
    def detect_anomalies(self, method: str = 'zscore') -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies in stock returns using specified method.
        
        Args:
            method: Anomaly detection method ('zscore', 'iqr', or 'rolling_zscore')
            
        Returns:
            Dictionary with ticker symbols as keys and anomaly DataFrames as values
        """
        logger.info(f"Detecting anomalies using {method} method")
        
        anomalies = {}
        
        for ticker, data in self.stock_data.items():
            if data.empty:
                anomalies[ticker] = pd.DataFrame()
                continue
            
            try:
                if method == 'zscore':
                    ticker_anomalies = self._detect_zscore_anomalies(data)
                elif method == 'iqr':
                    ticker_anomalies = self._detect_iqr_anomalies(data)
                elif method == 'rolling_zscore':
                    ticker_anomalies = self._detect_rolling_zscore_anomalies(data)
                else:
                    logger.error(f"Unknown anomaly detection method: {method}")
                    ticker_anomalies = pd.DataFrame()
                
                anomalies[ticker] = ticker_anomalies
                
                if not ticker_anomalies.empty:
                    logger.info(f"Found {len(ticker_anomalies)} anomalies for {ticker}")
                else:
                    logger.info(f"No anomalies found for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error detecting anomalies for {ticker}: {str(e)}")
                anomalies[ticker] = pd.DataFrame()
        
        self.anomalies = anomalies
        return anomalies
    
    def _detect_zscore_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method.
        """
        returns = data['Daily_Return']
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return pd.DataFrame()
        
        z_scores = np.abs((returns - mean_return) / std_return)
        anomaly_mask = z_scores > self.z_threshold
        
        anomalies = data[anomaly_mask].copy()
        anomalies['Z_Score'] = z_scores[anomaly_mask]
        
        return anomalies
    
    def _detect_iqr_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        """
        returns = data['Daily_Return']
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomaly_mask = (returns < lower_bound) | (returns > upper_bound)
        anomalies = data[anomaly_mask].copy()
        anomalies['IQR_Score'] = np.where(
            returns < lower_bound,
            (lower_bound - returns) / IQR,
            (returns - upper_bound) / IQR
        )[anomaly_mask]
        
        return anomalies
    
    def _detect_rolling_zscore_anomalies(self, data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Detect anomalies using rolling Z-score method.
        """
        returns = data['Daily_Return']
        rolling_mean = returns.rolling(window=window, min_periods=10).mean()
        rolling_std = returns.rolling(window=window, min_periods=10).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_scores = np.abs((returns - rolling_mean) / rolling_std)
        anomaly_mask = z_scores > self.z_threshold
        
        anomalies = data[anomaly_mask].copy()
        anomalies['Rolling_Z_Score'] = z_scores[anomaly_mask]
        
        return anomalies.dropna()
    
    def create_visualization(self, output_file: str = 'anomalies.png') -> None:
        """
        Create and save visualization of stock data with anomalies highlighted.
        
        Args:
            output_file: Output filename for the chart
        """
        logger.info(f"Creating visualization: {output_file}")
        
        # Count total tickers with data
        valid_tickers = [ticker for ticker, data in self.stock_data.items() if not data.empty]
        
        if not valid_tickers:
            logger.warning("No valid stock data to visualize")
            return
        
        # Create subplots
        n_tickers = len(valid_tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4 * n_tickers))
        
        if n_tickers == 1:
            axes = [axes]
        
        for i, ticker in enumerate(valid_tickers):
            data = self.stock_data[ticker]
            anomalies = self.anomalies.get(ticker, pd.DataFrame())
            
            ax = axes[i]
            
            # Plot daily returns
            ax.plot(data.index, data['Daily_Return'], label='Daily Returns', alpha=0.7, color='blue')
            
            # Highlight anomalies
            if not anomalies.empty:
                ax.scatter(anomalies.index, anomalies['Daily_Return'], 
                          color='red', s=50, label='Anomalies', zorder=5)
            
            ax.set_title(f'{ticker} - Daily Returns with Anomalies')
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add horizontal lines for reference
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add statistics text
            if not data.empty:
                mean_return = data['Daily_Return'].mean()
                std_return = data['Daily_Return'].std()
                ax.text(0.02, 0.98, f'Mean: {mean_return:.4f}\nStd: {std_return:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_file}")
    
    def generate_report(self, output_file: str = 'anomaly_report.json') -> None:
        """
        Generate and save JSON report of anomalies.
        
        Args:
            output_file: Output filename for the report
        """
        logger.info(f"Generating report: {output_file}")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'detection_parameters': {
                'z_threshold': self.z_threshold
            },
            'summary': {
                'total_tickers_analyzed': len(self.stock_data),
                'tickers_with_data': len([t for t, d in self.stock_data.items() if not d.empty]),
                'total_anomalies_found': sum(len(anomalies) for anomalies in self.anomalies.values())
            },
            'anomalies_by_ticker': {}
        }
        
        for ticker, anomalies in self.anomalies.items():
            if anomalies.empty:
                report['anomalies_by_ticker'][ticker] = {
                    'count': 0,
                    'anomalous_dates': []
                }
            else:
                anomaly_records = []
                for date, row in anomalies.iterrows():
                    record = {
                        'date': date.strftime('%Y-%m-%d'),
                        'daily_return': float(row['Daily_Return']),
                        'close_price': float(row['Close']),
                        'volume': int(row['Volume'])
                    }
                    
                    # Add score based on detection method
                    if 'Z_Score' in row:
                        record['z_score'] = float(row['Z_Score'])
                    elif 'IQR_Score' in row:
                        record['iqr_score'] = float(row['IQR_Score'])
                    elif 'Rolling_Z_Score' in row:
                        record['rolling_z_score'] = float(row['Rolling_Z_Score'])
                    
                    anomaly_records.append(record)
                
                report['anomalies_by_ticker'][ticker] = {
                    'count': len(anomalies),
                    'anomalous_dates': anomaly_records
                }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("ANOMALY DETECTION SUMMARY")
        print("="*50)
        print(f"Total tickers analyzed: {report['summary']['total_tickers_analyzed']}")
        print(f"Tickers with data: {report['summary']['tickers_with_data']}")
        print(f"Total anomalies found: {report['summary']['total_anomalies_found']}")
        print("\nAnomalies by ticker:")
        for ticker, info in report['anomalies_by_ticker'].items():
            print(f"  {ticker}: {info['count']} anomalies")
        print("="*50)


def main():
    """
    Main execution function demonstrating the stock anomaly detection system.
    """
    # Decision: data_selection - Which stocks to analyze?
    # Alternative 1: Popular tech stocks (AAPL, GOOGL, MSFT, TSLA)
    # Alternative 2: Diverse market sectors (tech, finance, energy, healthcare)
    # Chosen: Popular tech stocks for demonstration
    # Confidence: 0.85
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # Decision: parameter_tuning - What date range to analyze?
    # Alternative 1: Last 6 months for recent volatility
    # Alternative 2: Last 1 year for more comprehensive analysis
    # Chosen: Last 1 year for better statistical significance
    # Confidence: 0.90
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Decision: algorithm_selection - Which anomaly detection method?
    # Alternative 1: Z-score method (simple, interpretable)
    # Alternative 2: Rolling Z-score method (adaptive to trends)
    # Alternative 3: IQR method (robust to outliers)
    # Chosen: Z-score method for simplicity and interpretability
    # Confidence: 0.80
    detection_method = 'zscore'
    
    # Decision: parameter_tuning - What Z-score threshold?
    # Alternative 1: 2.0 (more sensitive, more anomalies)
    # Alternative 2: 2.5 (balanced sensitivity)
    # Alternative 3: 3.0 (less sensitive, fewer false positives)
    # Chosen: 2.5 for balanced detection
    # Confidence: 0.85
    z_threshold = 2.5
    
    print("Stock Anomaly Detection System")
    print("==============================")
    print(f"Analyzing tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Detection method: {detection_method}")
    print(f"Z-score threshold: {z_threshold}")
    print()
    
    try:
        # Initialize detector
        detector = StockAnomalyDetector(z_threshold=z_threshold)
        
        # Fetch stock data
        stock_data = detector.get_stock_data(tickers, start_date, end_date)
        
        # Check if we have any valid data
        valid_data = {k: v for k, v in stock_data.items() if not v.empty}
        if not valid_data:
            logger.error("No valid stock data retrieved. Exiting.")
            return
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(method=detection_method)
        
        # Create visualization
        detector.create_visualization('anomalies.png')
        
        # Generate report
        detector.generate_report('anomaly_report.json')
        
        print("\nAnalysis complete!")
        print("Generated files:")
        print("  - anomalies.png (visualization)")
        print("  - anomaly_report.json (detailed report)")
        print("  - stock_analysis.log (execution log)")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Check stock_analysis.log for details.")


if __name__ == '__main__':
    main()
