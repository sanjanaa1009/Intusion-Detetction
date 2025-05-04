import re
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Union, List

class UnknownThreatClassifier:
    """
    Detects and classifies unknown attack patterns not covered by UNSW dataset
    using Isolation Forest and advanced pattern matching.
    """
    
    def __init__(self):
        # Real-world attack patterns not in UNSW (expanded list)
        self.threat_patterns = {
            'Credential Stuffing': [
                (r'failed login for \w+ from \d+\.\d+\.\d+\.\d+', 3),
                (r'authentication attempt with \d+ passwords', 3)
            ],
            'API Abuse': [
                (r'api endpoint \S+ called \d+ times from \d+\.\d+\.\d+\.\d+', 2),
                (r'unusual api parameter: \S+=', 2)
            ],
            'Cloud Misconfig': [
                (r'public access enabled for \S+ bucket', 3),
                (r'security group \S+ allows 0\.0\.0\.0/0', 3)
            ],
            'Lateral Movement': [
                (r'connection from \d+\.\d+\.\d+\.\d+ to internal \S+', 2),
                (r'smb session established from \S+ to \S+', 2)
            ],
            'Cryptojacking': [
                (r'unexpected cpu spike from process \S+', 2),
                (r'crypto miner process detected', 3)
            ],
            'Supply Chain': [
                (r'dependency \S+ contains malicious code', 3),
                (r'package \S+ modified after installation', 2)
            ]
        }
        
        # Feature extraction pipeline
        self.feature_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 3),
                analyzer='word'
            )),
            ('detector', IsolationForest(
                n_estimators=150,
                contamination=0.05,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Thresholds
        self.pattern_threshold = 1.5  # Minimum pattern score
        self.anomaly_threshold = -0.6  # Isolation Forest score threshold

    def train(self, normal_logs: pd.DataFrame):
        """
        Train on normal logs to establish baseline patterns.
        
        Parameters:
            normal_logs: DataFrame with 'message' column containing clean logs
        """
        self.feature_pipeline.fit(normal_logs['message'].fillna(''))
        joblib.dump(self.feature_pipeline, 'unknown_threat_model.joblib')

    def detect(self, log_entry: Union[Dict, pd.Series]) -> Dict:
        """
        Detect unknown threats in a log entry.
        
        Returns:
            {
                'category': str,
                'confidence': float (0-3),
                'evidence': List[str],
                'is_unknown': bool
            }
        """
        if isinstance(log_entry, pd.Series):
            log_entry = log_entry.to_dict()
            
        message = log_entry.get('message', '')
        results = {
            'category': 'Normal',
            'confidence': 0,
            'evidence': [],
            'is_unknown': False
        }
        
        # Pattern Matching (Known Unknowns)
        for category, patterns in self.threat_patterns.items():
            matches = [
                pattern for pattern, score in patterns 
                if re.search(pattern, message, re.IGNORECASE)
            ]
            if matches:
                max_score = max(
                    score for pattern, score in patterns 
                    if pattern in matches
                )
                if max_score >= self.pattern_threshold:
                    results.update({
                        'category': category,
                        'confidence': min(max_score, 3),
                        'evidence': matches
                    })
        
        # Anomaly Detection (Unknown Unknowns)
        if results['category'] == 'Normal':
            anomaly_score = self.feature_pipeline.decision_function([message])[0]
            if anomaly_score < self.anomaly_threshold:
                results.update({
                    'category': 'Uncategorized Threat',
                    'confidence': self._score_to_confidence(anomaly_score),
                    'is_unknown': True,
                    'evidence': ['Anomalous pattern detected']
                })
        
        return results

    def _score_to_confidence(self, score: float) -> float:
        """Convert anomaly score to confidence level (0-3)"""
        return min(3, max(0, abs(score) * 2))

    def detect_batch(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple logs efficiently"""
        return pd.DataFrame([
            self.detect(log) 
            for _, log in logs_df.iterrows()
        ])

    @classmethod
    def load(cls, model_path: str):
        """Load trained classifier"""
        classifier = cls()
        classifier.feature_pipeline = joblib.load(model_path)
        return classifier