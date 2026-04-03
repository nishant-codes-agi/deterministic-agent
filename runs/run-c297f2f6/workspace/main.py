# Full content of main.py
import sys
import json
import traceback
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError as e:
    print(f"Error: Required package 'yfinance' not found. Please install it using: pip install yfinance")
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package 'pandas' not found. Please install it using: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    print(f"Error: Required package 'numpy' not found. Please install it using: pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Required package 'matplotlib' not found. Please install it using: pip install matplotlib")
    sys.exit(1)

# Constants
TICKER = 'AAPL'
ANOMALY_COLUMN = 'Close'
IQR_MULTIPLIER = 1.5

# Calculate date range: 1 year back from today
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

# Format dates for yfinance
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from yfinance with error handling."""
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}. Check the ticker symbol and date range.")
        
        # Handle MultiIndex columns if present (yfinance returns MultiIndex for multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex: keep only the first level (ticker) or specific columns
            # For single ticker, we can drop the first level
            data.columns = data.columns.droplevel(0) if data.columns.nlevels > 1 else data.columns
        
        print(f"Successfully fetched {len(data)} days of data.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Traceback:")
        traceback.print_exc()
        return None

def detect_anomalies_iqr(data, column, multiplier=1.5):
    """Detect anomalies using IQR method."""
    if data is None or data.empty:
        print("No data provided for anomaly detection.")
        return None, None, None, None
    
    # Extract the column values
    if column not in data.columns:
        print(f"Column '{column}' not found in data. Available columns: {list(data.columns)}")
        return None, None, None, None
    
    values = data[column]
    
    # Drop NaN values for calculation
    values_clean = values.dropna()
    if len(values_clean) == 0:
        print(f"No valid (non-NaN) values in column '{column}' for anomaly detection.")
        return None, None, None, None
    
    # Calculate percentiles
    try:
        q1 = np.percentile(values_clean, 25)
        q3 = np.percentile(values_clean, 75)
    except Exception as e:
        print(f"Error calculating percentiles: {e}")
        return None, None, None, None
    
    iqr = q3 - q1
    
    # Avoid division by zero or negative IQR (though unlikely for stock prices)
    if iqr <= 0:
        print(f"IQR is {iqr}, which is invalid for anomaly detection. Using standard deviation instead.")
        std = np.std(values_clean)
        lower_bound = np.mean(values_clean) - 3 * std
        upper_bound = np.mean(values_clean) + 3 * std
    else:
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
    
    # Identify anomalies
    anomalies = data[(values < lower_bound) | (values > upper_bound)]
    
    return anomalies, lower_bound, upper_bound, iqr

def calculate_returns(data, column='Close'):
    """Calculate daily returns."""
    if data is None or len(data) < 2:
        return pd.Series(dtype=float)
    
    if column not in data.columns:
        return pd.Series(dtype=float)
    
    # Calculate daily returns
    returns = data[column].pct_change() * 100  # Percentage
    return returns

def save_anomaly_report(anomalies, data, column, output_json='anomaly_report.json'):
    """Save anomaly report to JSON file."""
    if anomalies is None or anomalies.empty:
        report = {
            "ticker": TICKER,
            "period": {
                "start": START_DATE_STR,
                "end": END_DATE_STR
            },
            "anomaly_column": column,
            "total_data_points": len(data) if data is not None else 0,
            "anomaly_count": 0,
            "anomalies": []
        }
    else:
        # Calculate returns for anomaly days
        returns_series = calculate_returns(data, column)
        
        anomalies_list = []
        for idx, row in anomalies.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            value = float(row[column]) if pd.notna(row[column]) else None
            
            # Get return for this day
            return_val = None
            if idx in returns_series.index:
                return_val = float(returns_series.loc[idx]) if pd.notna(returns_series.loc[idx]) else None
            
            anomalies_list.append({
                "date": date_str,
                "value": value,
                "return_percent": return_val
            })
        
        report = {
            "ticker": TICKER,
            "period": {
                "start": START_DATE_STR,
                "end": END_DATE_STR
            },
            "anomaly_column": column,
            "total_data_points": len(data) if data is not None else 0,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies_list
        }
    
    try:
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Anomaly report saved to {output_json}")
    except Exception as e:
        print(f"Error saving JSON report: {e}")

def plot_anomalies(data, anomalies, column, lower_bound, upper_bound, output_png='anomalies.png'):
    """Plot data with anomalies highlighted."""
    if data is None or data.empty:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot the main data
    plt.plot(data.index, data[column], label=f'{column} Price', color='blue', alpha=0.7)
    
    # Plot anomalies if any
    if anomalies is not None and not anomalies.empty:
        plt.scatter(anomalies.index, anomalies[column], 
                   color='red', s=50, zorder=5, 
                   label=f'Anomalies ({len(anomalies)} found)')
    
    # Add threshold lines if available
    if lower_bound is not None and upper_bound is not None:
        plt.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label=f'Lower bound: {lower_bound:.2f}')
        plt.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7, label=f'Upper bound: {upper_bound:.2f}')
    
    plt.title(f'{TICKER} Stock Price Anomaly Detection ({START_DATE_STR} to {END_DATE_STR})')
    plt.xlabel('Date')
    plt.ylabel(f'{column} Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    try:
        plt.savefig(output_png, dpi=150)
        print(f"Plot saved to {output_png}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.show()

def print_anomalies_summary(anomalies, data, column, lower_bound, upper_bound, iqr):
    """Print summary of anomalies to console."""
    print("\n" + "="*60)
    print("ANOMALY DETECTION SUMMARY")
    print("="*60)
    print(f"Ticker: {TICKER}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Data points analyzed: {len(data) if data is not None else 0}")
    print(f"Column analyzed: {column}")
    
    if lower_bound is not None and upper_bound is not None and iqr is not None:
        print(f"IQR: {iqr:.4f}")
        print(f"Lower bound ({IQR_MULTIPLIER} * IQR): {lower_bound:.2f}")
        print(f"Upper bound ({IQR_MULTIPLIER} * IQR): {upper_bound:.2f}")
    
    if anomalies is None or anomalies.empty:
        print(f"Anomalies found: 0")
        print("No anomalies detected in the data.")
    else:
        print(f"Anomalies found: {len(anomalies)}")
        print("\nDetected anomalies:")
        print("-"*40)
        for idx, row in anomalies.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            value = row[column]
            print(f"Date: {date_str}, {column}: {value:.2f}")
    print("="*60 + "\n")

def main():
    """Main function to execute the anomaly detection pipeline."""
    print("Starting stock anomaly detection...")
    
    # Fetch data
    data = fetch_stock_data(TICKER, START_DATE_STR, END_DATE_STR)
    if data is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # Detect anomalies
    anomalies, lower_bound, upper_bound, iqr = detect_anomalies_iqr(
        data, ANOMALY_COLUMN, IQR_MULTIPLIER
    )
    
    # Print summary to console
    print_anomalies_summary(anomalies, data, ANOMALY_COLUMN, lower_bound, upper_bound, iqr)
    
    # Save report
    save_anomaly_report(anomalies, data, ANOMALY_COLUMN)
    
    # Plot and save visualization
    plot_anomalies(data, anomalies, ANOMALY_COLUMN, lower_bound, upper_bound)
    
    print("Anomaly detection completed successfully.")

if __name__ == "__main__":
    main()