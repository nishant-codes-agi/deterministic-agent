#!/usr/bin/env python3
"""
Data processor module for calculating indicators and preparing data for anomaly detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and indicator calculation."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw stock data and calculate indicators.
        
        Args:
            raw_data: Raw stock data from yfinance
            
        Returns:
            Processed DataFrame with additional indicators
        """
        try:
            logger.info("Processing stock data...")
            
            # Create a copy to avoid modifying original data
            data = raw_data.copy()
            
            # Calculate daily return (percentage change in Adjusted Close)
            data['Daily_Return'] = data['Close'].pct_change() * 100
            
            # Calculate rolling statistics for anomaly detection
            # Use 30-day rolling window for stability
            window = 30
            
            # Rolling mean and std for Daily Return
            data['Daily_Return_Mean'] = data['Daily_Return'].rolling(window=window, min_periods=1).mean()
            data['Daily_Return_Std'] = data['Daily_Return'].rolling(window=window, min_periods=1).std()
            
            # Rolling mean and std for Volume
            data['Volume_Mean'] = data['Volume'].rolling(window=window, min_periods=1).mean()
            data['Volume_Std'] = data['Volume'].rolling(window=window, min_periods=1).std()
            
            # Calculate Z-scores
            data['Daily_Return_ZScore'] = np.where(
                data['Daily_Return_Std'] > 0,
                (data['Daily_Return'] - data['Daily_Return_Mean']) / data['Daily_Return_Std'],
                0
            )
            
            data['Volume_ZScore'] = np.where(
                data['Volume_Std'] > 0,
                (data['Volume'] - data['Volume_Mean']) / data['Volume_Std'],
                0
            )
            
            # Handle any infinite or NaN values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values for the first few rows
            data = data.fillna(method='ffill')
            
            # Drop any remaining NaN rows
            data = data.dropna()
            
            logger.info(f"Processed {len(data)} rows of data")
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    def save_processed_data(self, data: pd.DataFrame, ticker: str) -> str:
        """
        Save processed data to file.
        
        Args:
            data: Processed DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            Path to saved file
        """
        file_path = self.data_dir / f"{ticker}_processed.csv"
        data.to_csv(file_path)
        logger.info(f"Processed data saved to {file_path}")
        return str(file_path)
