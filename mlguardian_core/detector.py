"""
Anomaly Detection for ML Models
Based on your proven Isolation Forests implementation from Sritech (65% faster inference)
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """
    Anomaly detection based on your proven Isolation Forests work.
    From Sritech internship: 65% faster inference time.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_baseline(self, data: np.ndarray):
        """
        Train on baseline healthy data.
        This is exactly what you did with sensor data at Sritech.
        """
        # Standardize data (important for Isolation Forests)
        scaled_data = self.scaler.fit_transform(data)
        
        # Train Isolation Forest (your proven method)
        self.isolation_forest.fit(scaled_data)
        self.is_trained = True
        print("Anomaly detector trained on baseline data.")
        
    def detect_anomalies(self, new_data: np.ndarray) -> dict:
        """
        Detect anomalies in new data.
        Returns detailed anomaly information.
        """
        if not self.is_trained:
            raise ValueError("Detector not trained. Call train_baseline first.")
            
        # Standardize new data using the fitted scaler
        scaled_new_data = self.scaler.transform(new_data)
        
        # Get anomaly predictions
        predictions = self.isolation_forest.predict(scaled_new_data)
        anomaly_scores = self.isolation_forest.decision_function(scaled_new_data)
        
        # Convert to boolean (True = anomaly)
        is_anomaly = predictions == -1
        
        return {
            'is_anomaly': is_anomaly.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_percentage': float(np.mean(is_anomaly) * 100)
        }

# Performance metrics calculator (based on your financial modeling experience)
class PerformanceMetrics:
    """
    Calculate model performance metrics.
    Based on your backtesting and financial modeling experience.
    """
    
    @staticmethod
    def calculate_latency_metrics(latencies: list) -> dict:
        """Calculate latency statistics."""
        if not latencies:
            return {}
            
        return {
            'mean_latency': float(np.mean(latencies)),
            'median_latency': float(np.median(latencies)),
            'p95_latency': float(np.percentile(latencies, 95)),
            'max_latency': float(np.max(latencies))
        }
    
    @staticmethod
    def calculate_prediction_stats(predictions: list) -> dict:
        """Calculate prediction distribution statistics."""
        if not predictions:
            return {}
            
        pred_array = np.array(predictions)
        return {
            'mean': float(np.mean(pred_array)),
            'std': float(np.std(pred_array)),
            'min': float(np.min(pred_array)),
            'max': float(np.max(pred_array))
        }