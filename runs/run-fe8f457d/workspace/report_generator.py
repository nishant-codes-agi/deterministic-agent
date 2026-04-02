#!/usr/bin/env python3
"""
Report generator module for creating HTML reports with visualizations.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging
from typing import List, Dict
from anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates HTML reports with interactive visualizations."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate_report(self, data: pd.DataFrame, ticker: str) -> str:
        """
        Generate a comprehensive HTML report with visualizations.
        
        Args:
            data: DataFrame with anomaly detection results
            ticker: Stock ticker symbol
            
        Returns:
            Path to generated HTML report
        """
        try:
            logger.info(f"Generating report for {ticker}...")
            
            # Create anomaly detector instance to get summary
            detector = AnomalyDetector()
            anomaly_summary = detector.get_anomaly_summary(data)
            
            # Create visualizations
            fig = self._create_visualizations(data, ticker, anomaly_summary)
            
            # Generate HTML content
            html_content = self._generate_html_content(fig, ticker, anomaly_summary, data)
            
            # Save report
            report_path = self.reports_dir / f"{ticker}_anomaly_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _create_visualizations(self, data: pd.DataFrame, ticker: str, anomaly_summary: List[Dict]) -> go.Figure:
        """
        Create interactive plotly visualizations.
        
        Args:
            data: DataFrame with stock data and anomaly flags
            ticker: Stock ticker symbol
            anomaly_summary: List of anomaly details
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{ticker} - Stock Price',
                f'{ticker} - Daily Returns (%)',
                f'{ticker} - Trading Volume'
            ),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Get anomalous dates
        anomalous_dates = data[data['Any_Anomaly']].index
        return_anomaly_dates = data[data['Return_Anomaly']].index
        volume_anomaly_dates = data[data['Volume_Anomaly']].index
        
        # Plot 1: Stock Price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Highlight anomalous days on price chart
        if len(anomalous_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalous_dates,
                    y=data.loc[anomalous_dates, 'Close'],
                    mode='markers',
                    name='Anomalous Days',
                    marker=dict(color='red', size=8, symbol='circle')
                ),
                row=1, col=1
            )
        
        # Plot 2: Daily Returns
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Daily_Return'],
                mode='lines',
                name='Daily Return',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        # Highlight return anomalies
        if len(return_anomaly_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=return_anomaly_dates,
                    y=data.loc[return_anomaly_dates, 'Daily_Return'],
                    mode='markers',
                    name='Return Anomalies',
                    marker=dict(color='red', size=8, symbol='triangle-up')
                ),
                row=2, col=1
            )
        
        # Plot 3: Volume
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Volume'],
                mode='lines',
                name='Volume',
                line=dict(color='orange', width=1)
            ),
            row=3, col=1
        )
        
        # Highlight volume anomalies
        if len(volume_anomaly_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=volume_anomaly_dates,
                    y=data.loc[volume_anomaly_dates, 'Volume'],
                    mode='markers',
                    name='Volume Anomalies',
                    marker=dict(color='red', size=8, symbol='diamond')
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text=f"{ticker} Stock Analysis - Anomaly Detection Report",
            title_x=0.5,
            showlegend=True
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
    
    def _generate_html_content(self, fig: go.Figure, ticker: str, 
                              anomaly_summary: List[Dict], data: pd.DataFrame) -> str:
        """
        Generate complete HTML content for the report.
        
        Args:
            fig: Plotly figure
            ticker: Stock ticker symbol
            anomaly_summary: List of anomaly details
            data: DataFrame with stock data
            
        Returns:
            Complete HTML content as string
        """
        # Convert plotly figure to HTML
        plot_html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
        # Calculate summary statistics
        total_days = len(data)
        anomalous_days = data['Any_Anomaly'].sum()
        return_anomalies = data['Return_Anomaly'].sum()
        volume_anomalies = data['Volume_Anomaly'].sum()
        
        # Generate anomaly table HTML
        anomaly_table_html = self._generate_anomaly_table(anomaly_summary)
        
        # Create complete HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{ticker} Anomaly Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .summary-card .number {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .anomaly-table {{
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .anomaly-type {{
            background-color: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 2px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{ticker} Stock Anomaly Detection Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Trading Days</h3>
                <div class="number">{total_days:,}</div>
            </div>
            <div class="summary-card">
                <h3>Anomalous Days</h3>
                <div class="number">{anomalous_days}</div>
            </div>
            <div class="summary-card">
                <h3>Return Anomalies</h3>
                <div class="number">{return_anomalies}</div>
            </div>
            <div class="summary-card">
                <h3>Volume Anomalies</h3>
                <div class="number">{volume_anomalies}</div>
            </div>
        </div>
        
        <div class="charts">
            {plot_html}
        </div>
        
        <div class="anomaly-table">
            <h2>Detected Anomalies</h2>
            {anomaly_table_html}
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_anomaly_table(self, anomaly_summary: List[Dict]) -> str:
        """
        Generate HTML table for anomaly summary.
        
        Args:
            anomaly_summary: List of anomaly details
            
        Returns:
            HTML table as string
        """
        if not anomaly_summary:
            return "<p>No anomalies detected in the analyzed period.</p>"
        
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price</th>
                    <th>Volume</th>
                    <th>Daily Return (%)</th>
                    <th>Anomaly Types</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for anomaly in anomaly_summary[:50]:  # Limit to first 50 anomalies
            anomaly_types_html = ""
            for anom_type in anomaly['Anomaly_Types']:
                anomaly_types_html += f'<span class="anomaly-type">{anom_type["Type"]} (Z: {anom_type["Z_Score"]})</span> '
            
            table_html += f"""
                <tr>
                    <td>{anomaly['Date']}</td>
                    <td>${anomaly['Close_Price']:,.2f}</td>
                    <td>{anomaly['Volume']:,}</td>
                    <td>{anomaly['Daily_Return']:.2f}%</td>
                    <td>{anomaly_types_html}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        if len(anomaly_summary) > 50:
            table_html += f"<p><em>Showing first 50 of {len(anomaly_summary)} total anomalies.</em></p>"
        
        return table_html
