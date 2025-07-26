"""
Test script to verify MLGuardian setup.
This will test your agent and detector components.
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from mlguardian_agent.agent import create_monitored_model
from mlguardian_core.detector import AnomalyDetector

def test_agent():
    """Test the MLGuardian agent with a simple model."""
    print("Testing MLGuardian Agent...")
    
    # Create a simple model (replace with your actual model)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Wrap with MLGuardian
    monitored_model = create_monitored_model(model, "test_linear_model")
    
    # Test predictions
    X_test = np.random.randn(10, 5)
    predictions = monitored_model.predict(X_test)
    
    print("Agent test completed. Check console output for monitoring data.")
    return monitored_model

def test_detector():
    """Test the anomaly detector with sample data."""
    print("\nTesting Anomaly Detector...")
    
    # Create sample healthy data
    healthy_data = np.random.randn(1000, 5)
    
    # Initialize detector (using your proven method)
    detector = AnomalyDetector(contamination=0.1)
    
    # Train on healthy data
    detector.train_baseline(healthy_data)
    
    # Test with new data
    test_data = np.random.randn(100, 5)
    results = detector.detect_anomalies(test_data)
    
    print(f"Anomaly percentage in test data: {results['anomaly_percentage']:.2f}%")
    print("Detector test completed.")
    
if __name__ == "__main__":
    test_agent()
    test_detector()