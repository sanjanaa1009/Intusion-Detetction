# train_model.py
import pandas as pd
from models.unknownThreat_detector import UnknownThreatClassifier  # Assuming your class is in this file
import numpy as np

def generate_sample_logs(num_samples=1000):
    """Generate synthetic normal logs for training"""
    normal_patterns = [
        "User login successful from 192.168.1.{}".format(i) for i in range(1, 101)
    ] + [
        "Connection established to database server",
        "Regular network traffic on port 443",
        "System backup completed successfully",
        "DNS query resolved for example.com",
        "HTTP 200 response for /index.html",
        "Scheduled task completed: system maintenance",
        "Firewall allowed outgoing connection",
        "Device connected to WiFi network"
    ]
    
    # Add some variations
    samples = []
    for _ in range(num_samples):
        template = np.random.choice(normal_patterns)
        # Add some random variations
        if "192.168.1" in template:
            template = template.replace("192.168.1", "10.0.0")
        samples.append(template)
    
    return pd.DataFrame({'message': samples})

if __name__ == "__main__":
    print("Generating training data...")
    train_data = generate_sample_logs(5000)  # Generate 5000 normal logs
    
    print("Training UnknownThreatClassifier...")
    classifier = UnknownThreatClassifier()
    classifier.train(train_data)
    
    print("Model trained and saved as 'unknown_threat_model.joblib'")