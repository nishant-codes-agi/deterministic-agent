#!/usr/bin/env python3
"""
FastAPI server for the stock anomaly detection API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import logging
from typing import Dict, Any

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Stock Anomaly Detection API",
        description="API for detecting anomalous trading days in stock data",
        version="1.0.0"
    )
    
    # Initialize components
    fetcher = DataFetcher()
    processor = DataProcessor()
    detector = AnomalyDetector()
    reporter = ReportGenerator()
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with API information."""
        return """
        <html>
            <head>
                <title>Stock Anomaly Detection API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    code { background-color: #e9ecef; padding: 2px 4px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Stock Anomaly Detection API</h1>
                    <p>This API provides stock anomaly detection using statistical methods.</p>
                    
                    <h2>Available Endpoints:</h2>
                    
                    <div class="endpoint">
                        <h3>GET /analyze/{ticker}</h3>
                        <p>Analyze a stock ticker for anomalies and generate a report.</p>
                        <p><strong>Example:</strong> <code>/analyze/TSLA</code></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>GET /report/{ticker}</h3>
                        <p>Get the HTML report for a previously analyzed ticker.</p>
                        <p><strong>Example:</strong> <code>/report/TSLA</code></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>GET /health</h3>
                        <p>Check API health status.</p>
                    </div>
                    
                    <h2>Usage:</h2>
                    <ol>
                        <li>Call <code>/analyze/TICKER</code> to process a stock and generate anomaly report</li>
                        <li>Call <code>/report/TICKER</code> to view the generated HTML report</li>
                    </ol>
                </div>
            </body>
        </html>
        """
    
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "message": "Stock Anomaly Detection API is running"}
    
    @app.get("/analyze/{ticker}")
    async def analyze_stock(ticker: str) -> Dict[str, Any]:
        """
        Analyze a stock ticker for anomalies.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
            
        Returns:
            Analysis results and report information
        """
        try:
            ticker = ticker.upper()
            logger.info(f"Starting analysis for {ticker}")
            
            # Fetch data
            try:
                raw_data = fetcher.fetch_stock_data(ticker, period="5y")
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except ConnectionError as e:
                raise HTTPException(status_code=503, detail=str(e))
            
            # Process data
            processed_data = processor.process_data(raw_data)
            
            # Detect anomalies
            anomaly_data = detector.detect_anomalies(processed_data)
            
            # Generate report
            report_path = reporter.generate_report(anomaly_data, ticker)
            
            # Get summary statistics
            total_days = len(anomaly_data)
            anomalous_days = anomaly_data['Any_Anomaly'].sum()
            return_anomalies = anomaly_data['Return_Anomaly'].sum()
            volume_anomalies = anomaly_data['Volume_Anomaly'].sum()
            
            # Get anomaly summary
            anomaly_summary = detector.get_anomaly_summary(anomaly_data)
            
            return {
                "ticker": ticker,
                "status": "success",
                "analysis_summary": {
                    "total_trading_days": int(total_days),
                    "anomalous_days": int(anomalous_days),
                    "return_anomalies": int(return_anomalies),
                    "volume_anomalies": int(volume_anomalies),
                    "anomaly_percentage": round((anomalous_days / total_days) * 100, 2)
                },
                "report_path": report_path,
                "report_url": f"/report/{ticker}",
                "recent_anomalies": anomaly_summary[:10]  # Last 10 anomalies
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.get("/report/{ticker}", response_class=HTMLResponse)
    async def get_report(ticker: str):
        """
        Get the HTML report for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            HTML report content
        """
        try:
            ticker = ticker.upper()
            report_path = Path("reports") / f"{ticker}_anomaly_report.html"
            
            if not report_path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Report not found for {ticker}. Please run analysis first using /analyze/{ticker}"
                )
            
            return FileResponse(
                path=report_path,
                media_type="text/html",
                filename=f"{ticker}_anomaly_report.html"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving report for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=f"Error serving report: {str(e)}")
    
    return app
