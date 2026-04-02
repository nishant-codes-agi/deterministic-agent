#!/usr/bin/env python3
"""
Main entry point for the stock anomaly detection API.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# Create necessary directories
for directory in ['data', 'reports', 'api', 'utils']:
    Path(directory).mkdir(exist_ok=True)

def install_missing_packages():
    """Install missing packages automatically."""
    required_packages = [
        'yfinance',
        'pandas',
        'numpy',
        'fastapi',
        'uvicorn',
        'plotly',
        'kaleido'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install missing packages before importing
install_missing_packages()

# Now import the modules
from data_fetcher import DataFetcher
from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector
from report_generator import ReportGenerator
from api_server import create_app

def main():
    """Main function to run the complete pipeline."""
    try:
        # Initialize components
        fetcher = DataFetcher()
        processor = DataProcessor()
        detector = AnomalyDetector()
        reporter = ReportGenerator()
        
        # Run pipeline for TSLA
        ticker = "TSLA"
        print(f"Processing {ticker}...")
        
        # Fetch data
        raw_data = fetcher.fetch_stock_data(ticker, period="5y")
        
        # Process data
        processed_data = processor.process_data(raw_data)
        
        # Detect anomalies
        anomaly_data = detector.detect_anomalies(processed_data)
        
        # Generate report
        report_path = reporter.generate_report(anomaly_data, ticker)
        print(f"Report generated: {report_path}")
        
        # Start API server
        print("Starting API server...")
        app = create_app()
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
