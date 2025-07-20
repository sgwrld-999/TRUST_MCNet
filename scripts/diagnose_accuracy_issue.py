#!/usr/bin/env python3
"""
Diagnostic script to identify and fix accuracy discrepancy issues in TRUST_MCNet.

This script helps debug the difference between client-reported accuracy (>80%) 
and global model accuracy (21%).
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trust_mcnet.utils.dataset_registry import DataManager
from src.trust_mcnet.partitioning import DirichletPartitioner
from src.trust_mcnet.partitioning import PartitionConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_data_distribution():
    """Analyze how data is distributed among clients vs global test."""
    
    # Load IoT dataset configuration
    config = {
        'name': 'iot_general',
        'path': './data/IoT_Datasets',
        'dataset_files': [
            "CIC_IOMT_2024_100_Samples.csv",
            "CIC_IoT_2023_100_Samples.csv", 
            "Edge_IIoT_100_Samples.csv",
            "IoT_23_100_Samples.csv",
            "MedBIoT_100_Samples.csv"
        ],
        'num_clients': 5,
        'eval_fraction': 0.2,
        'partitioning': 'dirichlet',
        'dirichlet_alpha': 1.0,
        'preprocessing': {
            'exclude_columns': [
                'timestamp', 'unique_id', 'orig_host', 'resp_host',
                'orig_port', 'resp_port', 'missed_bytes', 'tunnel_parents',
                'history', 'Label', 'Sub_Label'
            ]
        },
        'label_config': {
            'target_column': 'Label',
            'normal_labels': ['BenignTraffic', 'Benign', 'Mirai_BenignTraffic'],
            'auto_anomaly_detection': True
        }
    }
    
    print("=== TRUST_MCNet Accuracy Discrepancy Diagnosis ===\n")
    
    try:
        # Load data
        data_manager = DataManager(config)
        train_dataset, test_dataset = data_manager.load_datasets()
        
        print(f"✓ Successfully loaded datasets")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Global test samples: {len(test_dataset)}")
        
        # Analyze label distribution in global test set
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        test_label_counts = pd.Series(test_labels).value_counts()
        
        print(f"\n=== Global Test Set Analysis ===")
        print(f"Label distribution in global test set:")
        for label, count in test_label_counts.items():
            label_name = "Normal" if label == 0 else "Anomaly"
            percentage = (count / len(test_labels)) * 100
            print(f"  - {label_name} (Label {label}): {count} samples ({percentage:.1f}%)")
        
        # Partition training data among clients
        partition_config = PartitionConfig(alpha=config['dirichlet_alpha'])
        partitioner = DirichletPartitioner(partition_config)
        client_subsets = partitioner.partition(train_dataset, config['num_clients'])
        
        print(f"\n=== Client Data Distribution Analysis ===")
        print(f"Partitioning: {config['partitioning']} (alpha={config['dirichlet_alpha']})")
        
        total_train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        train_label_counts = pd.Series(total_train_labels).value_counts()
        
        print(f"\nTraining data label distribution:")
        for label, count in train_label_counts.items():
            label_name = "Normal" if label == 0 else "Anomaly"
            percentage = (count / len(total_train_labels)) * 100
            print(f"  - {label_name} (Label {label}): {count} samples ({percentage:.1f}%)")
        
        # Analyze each client's data distribution
        print(f"\n=== Per-Client Analysis ===")
        for client_id, subset in enumerate(client_subsets):
            client_labels = [subset[i][1] for i in range(len(subset))]
            client_label_counts = pd.Series(client_labels).value_counts()
            
            print(f"\nClient {client_id}: {len(subset)} samples")
            for label in [0, 1]:  # Normal, Anomaly
                count = client_label_counts.get(label, 0)
                percentage = (count / len(subset)) * 100 if len(subset) > 0 else 0
                label_name = "Normal" if label == 0 else "Anomaly"
                print(f"  - {label_name}: {count} samples ({percentage:.1f}%)")
        
        # Check data distribution similarity
        print(f"\n=== Distribution Similarity Analysis ===")
        
        # Calculate class balance in global test vs training
        test_anomaly_ratio = test_label_counts.get(1, 0) / len(test_labels)
        train_anomaly_ratio = train_label_counts.get(1, 0) / len(total_train_labels)
        
        print(f"Anomaly ratio in training data: {train_anomaly_ratio:.3f}")
        print(f"Anomaly ratio in global test: {test_anomaly_ratio:.3f}")
        print(f"Distribution mismatch: {abs(test_anomaly_ratio - train_anomaly_ratio):.3f}")
        
        if abs(test_anomaly_ratio - train_anomaly_ratio) > 0.1:
            print("⚠️  WARNING: Significant distribution mismatch between training and test data!")
            print("   This could explain the accuracy discrepancy.")
        else:
            print("✓ Training and test distributions are reasonably similar.")
        
        # Check if each client can handle both classes
        print(f"\n=== Client Capability Analysis ===")
        problematic_clients = 0
        for client_id, subset in enumerate(client_subsets):
            client_labels = [subset[i][1] for i in range(len(subset))]
            unique_labels = set(client_labels)
            
            if len(unique_labels) < 2:
                print(f"⚠️  Client {client_id}: Only has class {list(unique_labels)} - cannot learn both classes!")
                problematic_clients += 1
            else:
                print(f"✓ Client {client_id}: Has both classes - can learn complete task")
        
        if problematic_clients > 0:
            print(f"\n❌ ISSUE FOUND: {problematic_clients}/{len(client_subsets)} clients have incomplete class representation!")
            print("   This explains why global accuracy is poor - clients are learning partial tasks.")
            print("\n💡 SOLUTION: Use higher Dirichlet alpha (>= 1.0) or IID partitioning for better class distribution.")
        
        return {
            'train_size': len(train_dataset),
            'test_size': len(test_dataset), 
            'num_clients': len(client_subsets),
            'distribution_mismatch': abs(test_anomaly_ratio - train_anomaly_ratio),
            'problematic_clients': problematic_clients
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.exception("Full error details:")
        return None


def suggest_fixes(analysis_results: Dict):
    """Suggest specific fixes based on analysis results."""
    
    if analysis_results is None:
        return
    
    print(f"\n=== RECOMMENDED FIXES ===")
    
    # Fix 1: Data partitioning
    if analysis_results['problematic_clients'] > 0:
        print(f"\n1. 🔧 Fix Data Partitioning:")
        print(f"   Update config/dataset/iot_general.yaml:")
        print(f"   ```yaml")
        print(f"   partitioning: 'dirichlet'")
        print(f"   dirichlet_alpha: 2.0  # Higher alpha = more IID-like")
        print(f"   ```")
        print(f"   OR use IID partitioning for maximum mixing:")
        print(f"   ```yaml")
        print(f"   partitioning: 'iid'")
        print(f"   ```")
    
    # Fix 2: Feature preprocessing  
    print(f"\n2. 🧹 Clean Feature Set:")
    print(f"   Remove network-specific features that don't generalize:")
    print(f"   - orig_host, resp_host (IP addresses)")
    print(f"   - orig_port, resp_port (specific ports)")
    print(f"   - tunnel_parents, missed_bytes (network metadata)")
    
    # Fix 3: Trust mechanism tuning
    print(f"\n3. ⚙️  Tune Trust Mechanism:")
    print(f"   Lower trust threshold to include more client updates:")
    print(f"   ```yaml")
    print(f"   trust:")
    print(f"     threshold: 0.3  # Lower from 0.5")
    print(f"     use_dynamic_weights: true")
    print(f"   ```")
    
    # Fix 4: Validation
    print(f"\n4. ✅ Validate Fix:")
    print(f"   After changes, check that:")
    print(f"   - Each client has both normal and anomaly samples")
    print(f"   - Global accuracy improves significantly")
    print(f"   - Accuracy gap reduces to <20%")


if __name__ == "__main__":
    results = analyze_data_distribution()
    suggest_fixes(results)
    
    print(f"\n=== Next Steps ===")
    print(f"1. Apply the recommended configuration changes")
    print(f"2. Run your simulation again")
    print(f"3. Compare before/after accuracy metrics")
    print(f"4. Re-run this diagnostic if issues persist")
