# TSLA Stock Anomaly Detection Report

**Analysis Date:** 2026-04-03 11:31:08
**Period:** Last 14 trading days
**Z-Score Threshold:** ±2.0
**Data Source:** yfinance

## Summary Statistics

- **Total Trading Days Analyzed:** 13
- **Mean Daily Return:** -0.67%
- **Standard Deviation:** 3.07%
- **Minimum Daily Return:** -5.42%
- **Maximum Daily Return:** 4.64%

## Anomaly Detection Results

**Total Anomalies Detected:** 0

No anomalous trading days detected in the analyzed period.

This suggests that all daily returns were within the normal range 
(Z-score between -2.0 and +2.0).

## Methodology

This analysis uses Z-score normalization to identify anomalous trading days:

1. **Data Collection:** Historical stock data retrieved from Yahoo Finance
2. **Return Calculation:** Daily returns computed as (Close_t - Close_t-1) / Close_t-1
3. **Z-Score Calculation:** Z = (Return - Mean) / Standard Deviation
4. **Anomaly Detection:** Days with |Z-Score| > 2.0 flagged as anomalous

## Files Generated

- `anomalies.png` - Visualization of stock price and returns with anomalies highlighted
- `anomaly_report.json` - Detailed JSON report with all analysis data
- `anomaly_report.md` - This markdown summary report