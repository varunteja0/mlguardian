"""
MLGuardian Agent - Monitors ML Models in Production
Based on your experience with 20TB sensor data pipelines at Sritech
"""

import requests
import time
import numpy as np
from typing import Any

class MLGuardianAgent:
    """
    Agent to wrap and monitor any scikit-learn model.
    This builds on your experience with anomaly detection and model monitoring.
    """
    
    def __init__(self, model_name: str, core_api_url: str = "http://localhost:8000"):
        self.model_name = model_name
        self.core_api_url = core_api_url
        
    def wrap_model(self, model: Any) -> Any:
        """
        Wrap a scikit-learn model to add monitoring capabilities.
        Similar to how you wrapped sensor data processing in your Sritech project.
        """
        # Store original predict method
        original_predict = model.predict
        
        def monitored_predict(*args, **kwargs):
            # Capture input data
            inputs = args[0] if args else None
            
            # Get predictions
            start_time = time.time()
            predictions = original_predict(*args, **kwargs)
            latency = time.time() - start_time
            
            # Send monitoring data to core engine
            self._send_monitoring_data(inputs, predictions, latency)
            
            return predictions
        
        # Replace model's predict method
        model.predict = monitored_predict
        print(f"Model '{self.model_name}' is now being monitored by MLGuardian.")
        return model
    
    def _send_monitoring_data(self, inputs: Any, predictions: Any, latency: float):
        """
        Send monitoring data to the core engine.
        Based on your experience sending processed data in your Kafka pipelines.
        """
        try:
            monitoring_data = {
                'model_name': self.model_name,
                'timestamp': time.time(),
                'inputs_shape': str(np.array(inputs).shape) if inputs is not None else None,
                'predictions_shape': str(np.array(predictions).shape),
                'latency_seconds': latency
            }
            
            # In a real scenario, this would be a POST request to your FastAPI endpoint
            # For now, we'll just print it to simulate sending data
            print(f"Sending data to core engine: {monitoring_data}")
            # requests.post(f"{self.core_api_url}/monitor", json=monitoring_data)
            
        except Exception as e:
            print(f"Failed to send monitoring data: {e}")

# Convenience function to create monitored model
def create_monitored_model(model, model_name: str):
    """
    Convenience function to create a monitored model.
    """
    agent = MLGuardianAgent(model_name)
    return agent.wrap_model(model)