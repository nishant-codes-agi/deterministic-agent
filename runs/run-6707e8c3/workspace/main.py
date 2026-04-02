#!/usr/bin/env python3
"""
Stock Anomaly Detection API

This application pulls stock data from yfinance, detects anomalous trading days
using Z-score statistical method, generates HTML reports with visualizations,
and exposes results through a FastAPI endpoint.
"""

import logging
import socket
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Anomaly Detection API",
    description="Detect anomalous trading days using statistical methods",
    version="1.0.0"
)


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find an available port starting from start_port.
    
    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number
        
    Raises:
        RuntimeError: If no available port found
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch stock data from yfinance with error handling.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data (default: 5y)
        
    Returns:
        DataFrame with stock data
        
    Raises:
        ValueError: If ticker is invalid or no data available
    """
    try:
        logger.info(f"Fetching data for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data available for ticker: {ticker}")
            
        logger.info(f"Successfully fetched {len(data)} days of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise ValueError(f"Failed to fetch data for ticker {ticker}: {str(e)}")


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns and other features for anomaly detection.
    
    Args:
        data: Raw stock data DataFrame
        
    Returns:
        DataFrame with calculated features
    """
    try:
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate daily percentage returns
        df['Daily_Returns'] = df['Close'].pct_change() * 100
        
        # Use absolute volume (already in the data)
        df['Volume_Abs'] = df['Volume']
        
        # Drop first row with NaN daily returns
        df = df.dropna()
        
        logger.info(f"Calculated features for {len(df)} trading days")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating features: {str(e)}")
        raise ValueError(f"Failed to calculate features: {str(e)}")


def detect_anomalies(data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies using Z-score method.
    
    Args:
        data: DataFrame with calculated features
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly detection results
    """
    try:
        df = data.copy()
        
        # Calculate Z-scores for daily returns
        returns_mean = df['Daily_Returns'].mean()
        returns_std = df['Daily_Returns'].std()
        df['Z_Score_Returns'] = (df['Daily_Returns'] - returns_mean) / returns_std
        
        # Calculate Z-scores for volume
        volume_mean = df['Volume_Abs'].mean()
        volume_std = df['Volume_Abs'].std()
        df['Z_Score_Volume'] = (df['Volume_Abs'] - volume_mean) / volume_std
        
        # Identify anomalies (either returns OR volume exceeds threshold)
        df['Is_Anomaly'] = (
            (np.abs(df['Z_Score_Returns']) > threshold) |
            (np.abs(df['Z_Score_Volume']) > threshold)
        )
        
        anomaly_count = df['Is_Anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalous trading days")
        
        return df
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise ValueError(f"Failed to detect anomalies: {str(e)}")


def create_visualizations(data: pd.DataFrame, ticker: str) -> tuple:
    """
    Create interactive visualizations using Plotly.
    
    Args:
        data: DataFrame with anomaly detection results
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (price_plot_html, returns_plot_html, volume_plot_html)
    """
    try:
        # Prepare data for plotting
        normal_data = data[~data['Is_Anomaly']]
        anomaly_data = data[data['Is_Anomaly']]
        
        # 1. Price plot with anomalies highlighted
        fig_price = go.Figure()
        
        # Normal days
        fig_price.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data['Close'],
            mode='lines',
            name='Normal Days',
            line=dict(color='blue')
        ))
        
        # Anomalous days
        if not anomaly_data.empty:
            fig_price.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['Close'],
                mode='markers',
                name='Anomalous Days',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
        
        fig_price.update_layout(
            title=f'{ticker} Stock Price with Anomalies',
            xaxis_title='Date',
            yaxis_title='Close Price ($)',
            hovermode='x unified'
        )
        
        # 2. Daily Returns plot
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data['Daily_Returns'],
            mode='lines',
            name='Normal Days',
            line=dict(color='green')
        ))
        
        if not anomaly_data.empty:
            fig_returns.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['Daily_Returns'],
                mode='markers',
                name='Anomalous Days',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
        
        fig_returns.update_layout(
            title=f'{ticker} Daily Returns with Anomalies',
            xaxis_title='Date',
            yaxis_title='Daily Returns (%)',
            hovermode='x unified'
        )
        
        # 3. Volume plot
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data['Volume_Abs'],
            mode='lines',
            name='Normal Days',
            line=dict(color='purple')
        ))
        
        if not anomaly_data.empty:
            fig_volume.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data['Volume_Abs'],
                mode='markers',
                name='Anomalous Days',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
        
        fig_volume.update_layout(
            title=f'{ticker} Trading Volume with Anomalies',
            xaxis_title='Date',
            yaxis_title='Volume',
            hovermode='x unified'
        )
        
        # Convert to HTML
        price_html = fig_price.to_html(include_plotlyjs='cdn', div_id='price_plot')
        returns_html = fig_returns.to_html(include_plotlyjs='cdn', div_id='returns_plot')
        volume_html = fig_volume.to_html(include_plotlyjs='cdn', div_id='volume_plot')
        
        logger.info("Successfully created visualizations")
        return price_html, returns_html, volume_html
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise ValueError(f"Failed to create visualizations: {str(e)}")


def generate_anomaly_table(data: pd.DataFrame) -> str:
    """
    Generate HTML table of anomalous days.
    
    Args:
        data: DataFrame with anomaly detection results
        
    Returns:
        HTML string of the anomaly table
    """
    try:
        anomalies = data[data['Is_Anomaly']].copy()
        
        if anomalies.empty:
            return "<p>No anomalous trading days detected.</p>"
        
        # Select relevant columns and format
        table_data = anomalies[[
            'Close', 'Daily_Returns', 'Volume_Abs', 
            'Z_Score_Returns', 'Z_Score_Volume'
        ]].copy()
        
        # Format numeric columns
        table_data['Close'] = table_data['Close'].round(2)
        table_data['Daily_Returns'] = table_data['Daily_Returns'].round(2)
        table_data['Volume_Abs'] = table_data['Volume_Abs'].astype(int)
        table_data['Z_Score_Returns'] = table_data['Z_Score_Returns'].round(3)
        table_data['Z_Score_Volume'] = table_data['Z_Score_Volume'].round(3)
        
        # Rename columns for display
        table_data.columns = [
            'Close Price ($)', 'Daily Returns (%)', 'Volume', 
            'Z-Score Returns', 'Z-Score Volume'
        ]
        
        # Convert to HTML table
        html_table = table_data.to_html(
            classes='table table-striped table-bordered',
            table_id='anomaly_table',
            escape=False
        )
        
        logger.info(f"Generated anomaly table with {len(anomalies)} entries")
        return html_table
        
    except Exception as e:
        logger.error(f"Error generating anomaly table: {str(e)}")
        return f"<p>Error generating anomaly table: {str(e)}</p>"


def generate_html_report(ticker: str, data: pd.DataFrame, 
                        price_plot: str, returns_plot: str, 
                        volume_plot: str, anomaly_table: str) -> str:
    """
    Generate complete HTML report.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with results
        price_plot: HTML for price plot
        returns_plot: HTML for returns plot
        volume_plot: HTML for volume plot
        anomaly_table: HTML for anomaly table
        
    Returns:
        Complete HTML report string
    """
    try:
        # Calculate summary statistics
        total_days = len(data)
        anomaly_count = data['Is_Anomaly'].sum()
        anomaly_percentage = (anomaly_count / total_days) * 100
        
        date_range = f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Anomaly Detection Report - {ticker}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .plot-container {{ margin-bottom: 30px; }}
                .table-container {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Stock Anomaly Detection Report</h1>
                    <h2>Ticker: {ticker}</h2>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Analysis Period:</strong> {date_range}</p>
                </div>
                
                <div class="summary">
                    <h3>Summary Statistics</h3>
                    <div class="row">
                        <div class="col-md-3">
                            <strong>Total Trading Days:</strong> {total_days}
                        </div>
                        <div class="col-md-3">
                            <strong>Anomalous Days:</strong> {anomaly_count}
                        </div>
                        <div class="col-md-3">
                            <strong>Anomaly Rate:</strong> {anomaly_percentage:.2f}%
                        </div>
                        <div class="col-md-3">
                            <strong>Detection Method:</strong> Z-Score (threshold: 3.0)
                        </div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <h3>Stock Price Analysis</h3>
                    {price_plot}
                </div>
                
                <div class="plot-container">
                    <h3>Daily Returns Analysis</h3>
                    {returns_plot}
                </div>
                
                <div class="plot-container">
                    <h3>Trading Volume Analysis</h3>
                    {volume_plot}
                </div>
                
                <div class="table-container">
                    <h3>Detected Anomalous Trading Days</h3>
                    {anomaly_table}
                </div>
                
                <div class="mt-4">
                    <h4>Methodology</h4>
                    <p>This analysis uses Z-score based anomaly detection:</p>
                    <ul>
                        <li>Daily returns are calculated as percentage change in closing price</li>
                        <li>Z-scores are computed for both daily returns and trading volume</li>
                        <li>Days with absolute Z-score > 3.0 for either metric are flagged as anomalous</li>
                        <li>This corresponds to approximately 99.7% confidence interval</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        logger.info("Successfully generated HTML report")
        return html_template
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise ValueError(f"Failed to generate HTML report: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with API information.
    """
    return """
    <html>
        <head>
            <title>Stock Anomaly Detection API</title>
        </head>
        <body>
            <h1>Stock Anomaly Detection API</h1>
            <p>Welcome to the Stock Anomaly Detection API!</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/analyze_stock/TSLA">Analyze TSLA Stock</a></li>
                <li><a href="/analyze_stock/AAPL">Analyze AAPL Stock</a></li>
            </ul>
            <h2>Usage:</h2>
            <p>Visit <code>/analyze_stock/{ticker}</code> to analyze any stock ticker.</p>
        </body>
    </html>
    """


@app.get("/analyze_stock/{ticker}", response_class=HTMLResponse)
async def analyze_stock(ticker: str, period: str = "5y", threshold: float = 3.0):
    """
    Analyze stock for anomalous trading days and return HTML report.
    
    Args:
        ticker: Stock ticker symbol (e.g., TSLA, AAPL)
        period: Time period for analysis (default: 5y)
        threshold: Z-score threshold for anomaly detection (default: 3.0)
        
    Returns:
        HTML report with analysis results
    """
    try:
        logger.info(f"Starting analysis for ticker: {ticker}")
        
        # Validate inputs
        ticker = ticker.upper().strip()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        
        if threshold <= 0:
            raise HTTPException(status_code=400, detail="Threshold must be positive")
        
        # Step 1: Fetch stock data
        raw_data = fetch_stock_data(ticker, period)
        
        # Step 2: Calculate features
        processed_data = calculate_features(raw_data)
        
        # Step 3: Detect anomalies
        anomaly_data = detect_anomalies(processed_data, threshold)
        
        # Step 4: Create visualizations
        price_plot, returns_plot, volume_plot = create_visualizations(anomaly_data, ticker)
        
        # Step 5: Generate anomaly table
        anomaly_table = generate_anomaly_table(anomaly_data)
        
        # Step 6: Generate complete HTML report
        html_report = generate_html_report(
            ticker, anomaly_data, price_plot, returns_plot, volume_plot, anomaly_table
        )
        
        logger.info(f"Successfully completed analysis for {ticker}")
        return HTMLResponse(content=html_report)
        
    except ValueError as e:
        logger.error(f"Validation error for {ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error analyzing {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def main():
    """
    Main function to run the FastAPI application.
    """
    try:
        # Find available port
        port = find_available_port()
        
        logger.info("Starting Stock Anomaly Detection API...")
        logger.info(f"Access the API at: http://localhost:{port}")
        logger.info(f"API Documentation at: http://localhost:{port}/docs")
        logger.info(f"Example analysis: http://localhost:{port}/analyze_stock/TSLA")
        
        # Run the application
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False  # Disable reload to prevent port conflicts
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
