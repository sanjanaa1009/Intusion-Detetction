from models.anomaly_detection import AnomalyDetectionSystem
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Initialize the system
    ads = AnomalyDetectionSystem()
    
    # Load data
    train_df = pd.read_parquet("data/UNSW_NB15_training-set.parquet")
    test_df = pd.read_parquet("data/UNSW_NB15_testing-set.parquet")
    
    # Preprocess data
    train_df_clean = ads.preprocess(train_df)
    test_df_clean = ads.preprocess(test_df)
    
    # Add features
    train_df_clean = ads.add_features(train_df_clean)
    test_df_clean = ads.add_features(test_df_clean)
    
    # Align columns
    test_df_clean = test_df_clean.reindex(columns=train_df_clean.columns, fill_value=0)
    
    # Prepare data for binary model
    X_train_binary = train_df_clean[ads.features_to_keep]
    y_train_binary = train_df['label']
    X_test_binary = test_df_clean[ads.features_to_keep]
    y_test_binary = test_df['label']
    
    # Train binary model
    print("Training binary model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_binary, y_train_binary, test_size=0.1, random_state=42
    )
    ads.train_binary_model(X_train, y_train, X_val, y_val)
    
    # Evaluate binary model
    print("\nBinary Model Evaluation:")
    ads.evaluate_model(ads.binary_model, X_test_binary, y_test_binary)
    
    # Prepare data for multiclass model
    X_train_multi = train_df_clean[ads.features_to_keep].copy()
    X_train_multi['sbytes_stcpb'] = X_train_multi['sbytes'] * X_train_multi['stcpb']
    X_train_multi['synack_ackdat'] = X_train_multi['synack'] + X_train_multi['ackdat']
    y_train_multi = train_df['attack_cat']
    
    X_test_multi = test_df_clean[ads.features_to_keep].copy()
    X_test_multi['sbytes_stcpb'] = X_test_multi['sbytes'] * X_test_multi['stcpb']
    X_test_multi['synack_ackdat'] = X_test_multi['synack'] + X_test_multi['ackdat']
    y_test_multi = test_df['attack_cat']
    
    # Train multiclass model
    print("\nTraining multiclass model...")
    ads.train_multiclass_model(X_train_multi, y_train_multi)
    
    # Evaluate multiclass model
    print("\nMulticlass Model Evaluation:")
    ads.evaluate_model(ads.multiclass_model, X_test_multi, y_test_multi)
    
    # Save models
    ads.save_models()
    
    print("\nModels trained and saved successfully!")