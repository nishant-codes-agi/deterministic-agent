#!/usr/bin/env python3
"""
Anomaly detector module using Z-score based statistical method.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detects anomalies using Z-score based statistical method."""
    
    def __init__(self, return_threshold: float = 2.5, volume_threshold: float = 3.0):
        """
        Initialize anomaly detector with thresholds.
        
        Args:
            return_threshold: Z-score threshold for daily return anomalies
            volume_threshold: Z-score threshold for volume anomalies
        """
        self.return_threshold = return_threshold
        self.volume_threshold = volume_threshold
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the processed stock data.
        
        Args:
            data: Processed DataFrame with Z-scores
            
        Returns:
            DataFrame with anomaly flags and summary
        """
        try:
            logger.info("Detecting anomalies...")
            
            # Create a copy to avoid modifying original data
            result_data = data.copy()
            
            # Detect return anomalies
            result_data['Return_Anomaly'] = (
                np.abs(result_data['Daily_Return_ZScore']) > self.return_threshold
            )
            
            # Detect volume anomalies
            result_data['Volume_Anomaly'] = (
                np.abs(result_data['Volume_ZScore']) > self.volume_threshold
            )
            
            # Combined anomaly flag
            result_data['Any_Anomaly'] = (
                result_data['Return_Anomaly'] | result_data['Volume_Anomaly']
            )
            
            # Count anomalies
            return_anomalies = result_data['Return_Anomaly'].sum()
            volume_anomalies = result_data['Volume_Anomaly'].sum()
            total_anomalies = result_data['Any_Anomaly'].sum()
            
            logger.info(f"Detected {return_anomalies} return anomalies")
            logger.info(f"Detected {volume_anomalies} volume anomalies")
            logger.info(f"Total unique anomalous days: {total_anomalies}")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def get_anomaly_summary(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate a summary of detected anomalies.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            List of dictionaries containing anomaly details
        """
        anomalies = []
        
        # Get all anomalous days
        anomalous_days = data[data['Any_Anomaly']].copy()
        
        for date, row in anomalous_days.iterrows():
            anomaly_info = {
                'Date': date.strftime('%Y-%m-%d'),
                'Close_Price': round(row['Close'], 2),
                'Volume': int(row['Volume']),
                'Daily_Return': round(row['Daily_Return'], 2),
                'Anomaly_Types': []
            }
            
            if row['Return_Anomaly']:
                anomaly_info['Anomaly_Types'].append({
                    'Type': 'Daily Return',
                    'Z_Score': round(row['Daily_Return_ZScore'], 2),
                    'Value': round(row['Daily_Return'], 2),
                    'Threshold': self.return_threshold
                })
            
            if row['Volume_Anomaly']:
                anomaly_info['Anomaly_Types'].append({
                    'Type': 'Volume',
                    'Z_Score': round(row['Volume_ZScore'], 2),
                    'Value': int(row['Volume']),
                    'Threshold': self.volume_threshold
                })
            
            anomalies.append(anomaly_info)
        
        # Sort by date (most recent first)
        anomalies.sort(key=lambda x: x['Date'], reverse=True)
        
        return anomalies
