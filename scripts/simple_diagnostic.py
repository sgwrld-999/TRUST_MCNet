#!/usr/bin/env python3
"""
Simple diagnostic to check why global accuracy is 21%.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def analyze_simulation_data():
    """Analyze the data as it's processed in start_simulation.py"""
    
    print("=== TRUST_MCNet Data Distribution Analysis ===\n")
    
    # Load datasets exactly as start_simulation.py does
    dataset_files = {
        'CIC_IOMT_2024': 'data/IoT_Datasets/CIC_IOMT_2024_100_Samples.csv',
        'CIC_IoT_2023': 'data/IoT_Datasets/CIC_IoT_2023_100_Samples.csv',
        'Edge_IIoT': 'data/IoT_Datasets/Edge_IIoT_100_Samples.csv',
        'IoT_23': 'data/IoT_Datasets/IoT_23_100_Samples.csv',
        'MedBIoT': 'data/IoT_Datasets/MedBIoT_100_Samples.csv'
    }
    
    combined_data = []
    
    for name, file_path in dataset_files.items():
        if not Path(file_path).exists():
            print(f"❌ Missing file: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df['dataset_source'] = name
        combined_data.append(df)
        print(f"✓ Loaded {name}: {df.shape}")
        
        # Check label distribution per dataset
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            print(f"  Labels in {name}: {dict(label_counts)}")
    
    if not combined_data:
        print("❌ No data loaded!")
        return
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"\n✓ Combined dataset shape: {combined_df.shape}")
    
    # Check combined label distribution
    if 'Label' in combined_df.columns:
        print(f"\n=== Combined Label Distribution ===")
        label_counts = combined_df['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    # Preprocess exactly as start_simulation.py does
    try:
        # Handle labels
        label_encoder = LabelEncoder()
        combined_df['Label_Encoded'] = label_encoder.fit_transform(
            combined_df['Label'].astype(str)
        )
        
        print(f"\n=== Label Encoding ===")
        unique_labels = combined_df['Label'].unique()
        for i, label in enumerate(sorted(unique_labels)):
            encoded_value = label_encoder.transform([str(label)])[0]
            print(f"  '{label}' → {encoded_value}")
        
        # Select numeric features
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['Label_Encoded']]
        
        print(f"\n=== Feature Selection ===")
        print(f"  Total columns: {len(combined_df.columns)}")
        print(f"  Numeric columns: {len(numeric_columns)}")
        print(f"  Feature columns: {len(feature_columns)}")
        print(f"  Features: {feature_columns[:5]}...")  # Show first 5
        
        # Feature scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(combined_df[feature_columns])
        y = combined_df['Label_Encoded'].values
        
        print(f"\n=== Processed Data ===")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Label distribution: {np.bincount(y)}")
        
        # Train-test split (same as simulation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n=== Train-Test Split ===")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Train labels: {np.bincount(y_train)}")
        print(f"  Test labels: {np.bincount(y_test)}")
        
        # Simulate federated client partitioning
        print(f"\n=== Client Data Partitioning (IID) ===")
        num_clients = 5
        samples_per_client = len(X_train) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:  # Last client gets remaining samples
                end_idx = len(X_train)
            else:
                end_idx = (i + 1) * samples_per_client
                
            client_X = X_train[start_idx:end_idx]
            client_y = y_train[start_idx:end_idx]
            client_labels = np.bincount(client_y, minlength=2)
            
            print(f"  Client {i}: {len(client_X)} samples, labels: {client_labels}")
        
        # Test with a proper ML model
        print(f"\n=== ML Model Testing ===")
        
        # Train a logistic regression model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        # Test on global test set
        y_pred = lr_model.predict(X_test)
        global_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Global test accuracy (LogisticRegression): {global_accuracy:.3f}")
        
        # Test on each client's data separately
        print(f"\n=== Per-Client Accuracy Analysis ===")
        client_accuracies = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:
                end_idx = len(X_train)
            else:
                end_idx = (i + 1) * samples_per_client
                
            client_X = X_train[start_idx:end_idx]
            client_y = y_train[start_idx:end_idx]
            
            # Train on client data
            client_model = LogisticRegression(random_state=42, max_iter=1000)
            client_model.fit(client_X, client_y)
            
            # Test on client's own data (like simulation does)
            client_pred = client_model.predict(client_X)
            client_accuracy = accuracy_score(client_y, client_pred)
            client_accuracies.append(client_accuracy)
            
            print(f"  Client {i} local accuracy: {client_accuracy:.3f}")
        
        avg_client_accuracy = np.mean(client_accuracies)
        print(f"\n=== RESULT COMPARISON ===")
        print(f"  Average client accuracy: {avg_client_accuracy:.3f}")
        print(f"  Global model accuracy: {global_accuracy:.3f}")
        print(f"  Accuracy gap: {abs(avg_client_accuracy - global_accuracy):.3f}")
        
        if global_accuracy < 0.5:
            print(f"\n❌ ISSUE DETECTED: Global accuracy is very low!")
            print(f"   This suggests a fundamental data or model issue.")
            print(f"   Check if labels are properly encoded and balanced.")
        elif abs(avg_client_accuracy - global_accuracy) > 0.3:
            print(f"\n❌ ISSUE DETECTED: Large accuracy gap!")
            print(f"   This suggests data distribution problems.")
        else:
            print(f"\n✅ Results look reasonable!")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_simulation_data()
