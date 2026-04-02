#!/usr/bin/env python3
"""
Stock Anomaly Detection API

This application pulls stock data from yfinance, detects anomalous trading days
using Z-score statistical method, and provides results through a FastAPI endpoint.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Anomaly Detection API",
    description="Detect anomalous trading days using statistical methods",
    version="1.0.0"
)

class StockAnomalyDetector:
    """Main class for stock anomaly detection and analysis."""
    
    def __init__(self, z_score_threshold: float = 3.0):
        self.z_score_threshold = z_score_threshold
        
    def fetch_stock_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        Fetch stock data from yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data (default: 5y)
            
        Returns:
            DataFrame with stock data
            
        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            logger.info(f"Successfully fetched {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Failed to fetch data for ticker {ticker}: {str(e)}")
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for anomaly detection.
        
        Args:
            data: Raw stock data
            
        Returns:
            DataFrame with calculated features
        """
        df = data.copy()
        
        # Calculate daily returns
        df['Daily_Returns'] = df['Close'].pct_change() * 100
        
        # Use volume as is (already absolute)
        df['Volume_Normalized'] = df['Volume']
        
        # Drop first row with NaN returns
        df = df.dropna()
        
        logger.info(f"Calculated features for {len(df)} trading days")
        return df
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method.
        
        Args:
            data: DataFrame with calculated features
            
        Returns:
            DataFrame with anomaly detection results
        """
        df = data.copy()
        
        # Calculate Z-scores for returns and volume
        returns_mean = df['Daily_Returns'].mean()
        returns_std = df['Daily_Returns'].std()
        df['Returns_ZScore'] = (df['Daily_Returns'] - returns_mean) / returns_std
        
        volume_mean = df['Volume_Normalized'].mean()
        volume_std = df['Volume_Normalized'].std()
        df['Volume_ZScore'] = (df['Volume_Normalized'] - volume_mean) / volume_std
        
        # Identify anomalies (either returns OR volume exceeds threshold)
        df['Is_Anomaly'] = (
            (np.abs(df['Returns_ZScore']) > self.z_score_threshold) |
            (np.abs(df['Volume_ZScore']) > self.z_score_threshold)
        )
        
        anomaly_count = df['Is_Anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalous trading days")
        
        return df
    
    def create_visualizations(self, data: pd.DataFrame, ticker: str) -> dict:
        """
        Create interactive visualizations using Plotly.
        
        Args:
            data: DataFrame with anomaly detection results
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing HTML plots
        """
        # Separate normal and anomalous days
        normal_days = data[~data['Is_Anomaly']]
        anomaly_days = data[data['Is_Anomaly']]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{ticker} Stock Price',
                f'{ticker} Daily Returns (%)',
                f'{ticker} Trading Volume'
            ),
            vertical_spacing=0.08
        )
        
        # Price plot
        fig.add_trace(
            go.Scatter(
                x=normal_days.index,
                y=normal_days['Close'],
                mode='lines',
                name='Normal Days',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_days.index,
                y=anomaly_days['Close'],
                mode='markers',
                name='Anomalous Days',
                marker=dict(color='red', size=6, symbol='diamond')
            ),
            row=1, col=1
        )
        
        # Returns plot
        fig.add_trace(
            go.Scatter(
                x=normal_days.index,
                y=normal_days['Daily_Returns'],
                mode='lines',
                name='Normal Returns',
                line=dict(color='green', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_days.index,
                y=anomaly_days['Daily_Returns'],
                mode='markers',
                name='Anomalous Returns',
                marker=dict(color='red', size=6, symbol='diamond'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Volume plot
        fig.add_trace(
            go.Scatter(
                x=normal_days.index,
                y=normal_days['Volume_Normalized'],
                mode='lines',
                name='Normal Volume',
                line=dict(color='purple', width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_days.index,
                y=anomaly_days['Volume_Normalized'],
                mode='markers',
                name='Anomalous Volume',
                marker=dict(color='red', size=6, symbol='diamond'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Stock Anomaly Analysis for {ticker}",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return {
            'main_plot': fig.to_html(include_plotlyjs='cdn', div_id='main-plot')
        }
    
    def generate_summary_table(self, data: pd.DataFrame) -> str:
        """
        Generate HTML summary table of anomalous days.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            HTML string containing the summary table
        """
        anomalies = data[data['Is_Anomaly']].copy()
        
        if anomalies.empty:
            return "<p>No anomalous trading days detected.</p>"
        
        # Format the data for display
        anomalies = anomalies[[
            'Close', 'Daily_Returns', 'Volume_Normalized', 
            'Returns_ZScore', 'Volume_ZScore'
        ]].round(4)
        
        anomalies.index = anomalies.index.strftime('%Y-%m-%d')
        
        # Create HTML table
        table_html = """
        <table class="table table-striped table-hover">
            <thead class="table-dark">
                <tr>
                    <th>Date</th>
                    <th>Close Price ($)</th>
                    <th>Daily Returns (%)</th>
                    <th>Volume</th>
                    <th>Returns Z-Score</th>
                    <th>Volume Z-Score</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for date, row in anomalies.iterrows():
            table_html += f"""
                <tr>
                    <td>{date}</td>
                    <td>{row['Close']:.2f}</td>
                    <td>{row['Daily_Returns']:.2f}</td>
                    <td>{row['Volume_Normalized']:,.0f}</td>
                    <td>{row['Returns_ZScore']:.2f}</td>
                    <td>{row['Volume_ZScore']:.2f}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def generate_html_report(self, data: pd.DataFrame, plots: dict, ticker: str) -> str:
        """
        Generate complete HTML report.
        
        Args:
            data: DataFrame with analysis results
            plots: Dictionary containing plot HTML
            ticker: Stock ticker symbol
            
        Returns:
            Complete HTML report string
        """
        anomaly_count = data['Is_Anomaly'].sum()
        total_days = len(data)
        anomaly_percentage = (anomaly_count / total_days) * 100
        
        summary_table = self.generate_summary_table(data)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Anomaly Analysis - {ticker}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .container {{ max-width: 1200px; }}
                .alert-info {{ background-color: #e3f2fd; border-color: #2196f3; }}
                .table {{ font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <div class="row">
                    <div class="col-12">
                        <h1 class="text-center mb-4">Stock Anomaly Detection Report</h1>
                        <h2 class="text-center text-muted mb-4">{ticker}</h2>
                        
                        <div class="alert alert-info" role="alert">
                            <h4 class="alert-heading">Analysis Summary</h4>
                            <p><strong>Total Trading Days Analyzed:</strong> {total_days:,}</p>
                            <p><strong>Anomalous Days Detected:</strong> {anomaly_count} ({anomaly_percentage:.2f}%)</p>
                            <p><strong>Detection Method:</strong> Z-Score (threshold: {self.z_score_threshold})</p>
                            <p><strong>Analysis Period:</strong> Last 5 years</p>
                            <hr>
                            <p class="mb-0">Anomalies are detected when either daily returns or trading volume 
                            have a Z-score with absolute value greater than {self.z_score_threshold}.</p>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <h3>Interactive Visualizations</h3>
                                {plots['main_plot']}
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <h3>Anomalous Trading Days</h3>
                                {summary_table}
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <p class="text-muted text-center">
                                    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def analyze_stock(self, ticker: str) -> str:
        """
        Complete analysis pipeline for a stock ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            HTML report string
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            # Fetch data
            raw_data = self.fetch_stock_data(ticker)
            
            # Calculate features
            data_with_features = self.calculate_features(raw_data)
            
            # Detect anomalies
            analyzed_data = self.detect_anomalies(data_with_features)
            
            # Create visualizations
            plots = self.create_visualizations(analyzed_data, ticker)
            
            # Generate report
            html_report = self.generate_html_report(analyzed_data, plots, ticker)
            
            return html_report
            
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Analysis failed: {str(e)}")

# Initialize detector
detector = StockAnomalyDetector()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Stock Anomaly Detection API</title>
        </head>
        <body>
            <h1>Stock Anomaly Detection API</h1>
            <p>Use the following endpoints:</p>
            <ul>
                <li><a href="/analyze_stock/TSLA">/analyze_stock/TSLA</a> - Analyze Tesla stock</li>
                <li><a href="/analyze_stock/AAPL">/analyze_stock/AAPL</a> - Analyze Apple stock</li>
                <li><a href="/analyze_stock/MSFT">/analyze_stock/MSFT</a> - Analyze Microsoft stock</li>
            </ul>
            <p>Replace the ticker symbol with any valid stock ticker.</p>
        </body>
    </html>
    """

@app.get("/analyze_stock/{ticker}", response_class=HTMLResponse)
async def analyze_stock_endpoint(ticker: str):
    """
    Analyze a stock ticker and return HTML report.
    
    Args:
        ticker: Stock ticker symbol (e.g., TSLA, AAPL, MSFT)
        
    Returns:
        HTML report with anomaly analysis
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Validate ticker format
        ticker = ticker.upper().strip()
        if not ticker or len(ticker) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Invalid ticker format. Please provide a valid stock ticker."
            )
        
        logger.info(f"Starting analysis for ticker: {ticker}")
        
        # Perform analysis
        html_report = detector.analyze_stock(ticker)
        
        logger.info(f"Analysis completed successfully for {ticker}")
        return html_report
        
    except ValueError as e:
        logger.error(f"Analysis error for {ticker}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during analysis: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    # Test the detector with TSLA
    print("Testing Stock Anomaly Detector...")
    try:
        test_report = detector.analyze_stock("TSLA")
        print("✓ Analysis completed successfully")
        print(f"✓ Report generated ({len(test_report)} characters)")
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    # Start the API server
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)