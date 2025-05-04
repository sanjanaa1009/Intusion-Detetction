import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
# At the start of your script
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

class AnomalyDetectionSystem:
    def __init__(self):
        self.binary_model = None
        self.multiclass_model = None
        self.features_to_keep = [
            'sbytes', 'stcpb', 'synack', 'ackdat', 'dtcpb',
            'bytes_ratio', 'jit_ratio', 'dur', 'pkt_time_ratio',
            'tcprtt', 'sload', 'dbytes', 'dload', 'sinpkt',
            'sjit', 'djit', 'rate', 'load_ratio', 'dinpkt',
            'response_body_len'
        ]
        self.train_df = None
        self.test_df = None
    
    def load_data(self, train_path, test_path):
        """Load the training and testing datasets"""
        self.train_df = pd.read_parquet(train_path)
        self.test_df = pd.read_parquet(test_path)
        print(f"Training data loaded: {self.train_df.shape}")
        print(f"Testing data loaded: {self.train_df.shape}")
    
    def preprocess(self, df):
        """Preprocess the input dataframe"""
        df = df.copy()

        # Convert all columns (except categorical ones) to numeric
        for col in df.columns:
            if col not in ['proto', 'state', 'label', 'attack_cat']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop fully null columns
        null_cols = df.columns[df.isna().all()]
        if len(null_cols):
            print(f"Dropping fully null columns: {list(null_cols)}")
        df.drop(columns=null_cols, inplace=True)

        # Drop rows with NaN in crucial numeric fields
        crucial_cols = ['dur', 'sbytes', 'dbytes', 'rate']
        df.dropna(subset=crucial_cols, inplace=True)

        # Convert categorical columns to string and one-hot encode
        cat_cols = ['proto', 'state']
        df[cat_cols] = df[cat_cols].astype(str)
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df
    
    def add_features(self, df):
        """Add engineered features"""
        df = df.copy()
        df['bytes_ratio'] = (df['sbytes'] + 1) / (df['dbytes'] + 1)
        df['load_ratio'] = (df['sload'] + 1) / (df['dload'] + 1)
        df['pkt_time_ratio'] = (df['sinpkt'] + 1) / (df['dinpkt'] + 1)
        df['jit_ratio'] = (df['sjit'] + 1) / (df['djit'] + 1)
        
        # Add interaction features for multiclass model
        if all(col in df.columns for col in ['sbytes', 'stcpb', 'synack', 'ackdat']):
            df['sbytes_stcpb'] = df['sbytes'] * df['stcpb']
            df['synack_ackdat'] = df['synack'] + df['ackdat']
        
        return df
    
    def train_binary_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train the binary classification model"""
       
        # Your original:
        model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.015,
            num_leaves=80,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            sverbosity=-1
        )
        callbacks = [
            lgb.early_stopping(100, verbose=1),
            lgb.log_evaluation(50)
        ]
    
        if X_val is not None and y_val is not None:
            model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]  # Your original early stopping
            )
        else:
            model.fit(X_train, y_train)
    
        self.binary_model = model
        return model
         
    def train_multiclass_model(self, X_train, y_train, X_val=None, y_val=None):
    
    # 1. Optimized SMOTE sampling (reduce sample size if needed)
        sm = SMOTE(
        random_state=42,
        sampling_strategy='not majority',  # Only resample minority classes
        k_neighbors=3  # Reduce from default 5 for faster computation
      )
    
    # Only resample if classes are imbalanced
        if len(np.unique(y_train)) > 2:  # For multiclass only
            X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train.copy(), y_train.copy()

    # 2. Optimized LightGBM parameters
            model = lgb.LGBMClassifier(
        n_estimators=500,  # Reduced from 1500 (use early stopping)
        learning_rate=0.05,  # Increased from 0.015
        num_leaves=31,  # Reduced from 80 (faster training)
        max_depth=7,  # Reduced from 9
        subsample=0.8,  # Slightly reduced from 0.85
        colsample_bytree=0.8,  # Slightly reduced from 0.85
        reg_alpha=0.1,
        reg_lambda=0.2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        objective='multiclass',  # Explicit objective
        metric='multi_logloss',  # Proper multiclass metric
        boosting_type='gbdt'  # Explicit type
            )

    # 3. Optimized training with early stopping
        callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=1),  # Stop if no improvement
        lgb.log_evaluation(50)  # Print every 50 iterations
        ]

        model.fit(
            X_resampled, y_resampled,
         eval_set=[(X_val, y_val)] if X_val is not None else None,
            callbacks=callbacks
        )
    
        self.multiclass_model = model
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
        print(f"\n{'='*50}")
        print("Detailed Evaluation Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Plot feature importance
        lgb.plot_importance(model, max_num_features=20)
        plt.show()
    
    def save_models(self, path_prefix='anomaly_detection_models'):
        """Save both models to disk"""
        if self.binary_model:
            joblib.dump(self.binary_model, f'{path_prefix}_binary.joblib')
        if self.multiclass_model:
            joblib.dump(self.multiclass_model, f'{path_prefix}_multiclass.joblib')

    def load_models(self, path_prefix='anomaly_detection_models'):
        """Load models from disk"""
        self.binary_model = joblib.load(f'{path_prefix}_binary.joblib')
        self.multiclass_model = joblib.load(f'{path_prefix}_multiclass.joblib')
    
    def predict(self, df, model_type='binary'):
        """
        Make predictions on new data
        model_type: 'binary' or 'multiclass'
        """
        # Preprocess and add features
        processed_df = self.preprocess(df)
        processed_df = self.add_features(processed_df)
        
        # Select features
        features = [f for f in self.features_to_keep if f in processed_df.columns]
        X = processed_df[features]
        
        # Make predictions
        if model_type == 'binary' and self.binary_model:
            return self.binary_model.predict(X)
        elif model_type == 'multiclass' and self.multiclass_model:
            return self.multiclass_model.predict(X)
        else:
            raise ValueError("Model not available or invalid model_type")