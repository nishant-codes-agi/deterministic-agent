#!/usr/bin/env python3
"""
Data fetcher module for downloading stock data from yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles fetching stock data from yfinance."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def fetch_stock_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        Fetch stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA')
            period: Time period for data (e.g., '5y', '1y', '6mo')
            
        Returns:
            DataFrame with stock data
            
        Raises:
            ValueError: If ticker is invalid or no data found
            ConnectionError: If network issues occur
        """
        try:
            logger.info(f"Fetching data for {ticker} with period {period}")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Download historical data
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Save raw data
            file_path = self.data_dir / f"{ticker}_raw_{period}.csv"
            data.to_csv(file_path)
            logger.info(f"Raw data saved to {file_path}")
            
            return data
            
        except Exception as e:
            if "No data found" in str(e) or "Invalid ticker" in str(e):
                raise ValueError(f"Invalid ticker or no data available: {ticker}")
            else:
                raise ConnectionError(f"Network error fetching data for {ticker}: {e}")
    
    def load_cached_data(self, ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Load previously cached data if available.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data
            
        Returns:
            DataFrame if cached data exists, None otherwise
        """
        file_path = self.data_dir / f"{ticker}_raw_{period}.csv"
        
        if file_path.exists():
            try:
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data from {file_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        return None
