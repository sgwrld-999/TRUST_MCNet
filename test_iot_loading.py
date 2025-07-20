#!/usr/bin/env python3
"""
Test the IoT dataset loading with adaptive feature selection to see what's happening.
"""

import sys
import os
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trust_mcnet.utils.dataset_registry import DatasetRegistry
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_iot_dataset_loading():
    """Test the full IoT dataset loading process."""
    
    config = {
        'name': 'iot_general',
        'path': './data/IoT_Datasets',
        'dataset_files': [
            'CIC_IOMT_2024_100_Samples.csv',
            'CIC_IoT_2023_100_Samples.csv', 
            'Edge_IIoT_100_Samples.csv',
            'IoT_23_100_Samples.csv',
            'MedBIoT_100_Samples.csv'
        ],
        'num_clients': 5,
        'eval_fraction': 0.1,  # Smaller test set to avoid stratification issues
        'label_config': {
            'target_column': 'Label',
            'normal_labels': ['BenignTraffic', 'Benign', 'Mirai_BenignTraffic'],
            'anomaly_labels': ['DDoS', 'BruteForce', 'Recon', 'Mirai_DDoS', 'Ransomware', 'Malicious', 'bashlite', 'Mirai', 'web'],
            'auto_anomaly_detection': True,
            'balance_classes': True,
            'balancing_method': 'undersample'
        },
        'preprocessing': {
            'handle_missing_values': True,
            'missing_value_strategy': 'median',
            'auto_remove_redundant': True,
            'remove_timestamps': True,
            'remove_addresses': True,
            'adaptive_feature_selection': True,
            'feature_selection': {
                'remove_metadata': True,
                'remove_timestamps': True,
                'remove_addresses': True,
                'remove_constant': True,
                'remove_high_cardinality': True,
                'high_cardinality_threshold': 0.9,
                'remove_correlated': True,
                'variance_filtering': True,
                'variance_threshold': 0.01,
                'max_features': 15
            },
            'standardization': True,
            'normalization_method': 'standard',
            'scale_features': True
        }
    }
    
    logger.info("Testing IoT dataset loading with adaptive feature selection...")
    
    try:
        # Get the dataset loader
        loader = DatasetRegistry.get_loader(config['name'])
        logger.info(f"Got loader: {type(loader)}")
        
        # Load the dataset
        train_dataset, test_dataset = loader.load(config)
        
        logger.info(f"Loaded datasets:")
        logger.info(f"  Train dataset: {len(train_dataset)} samples")
        logger.info(f"  Test dataset: {len(test_dataset)} samples")
        
        # Check data shapes
        if hasattr(train_dataset, 'tensors'):
            X_train, y_train = train_dataset.tensors
            logger.info(f"  Train data shape: {X_train.shape}")
            logger.info(f"  Train labels shape: {y_train.shape}")
        
        if hasattr(test_dataset, 'tensors'):
            X_test, y_test = test_dataset.tensors
            logger.info(f"  Test data shape: {X_test.shape}")
            logger.info(f"  Test labels shape: {y_test.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load IoT dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_iot_dataset_loading()
    if success:
        logger.info("✅ IoT dataset loading test completed!")
    else:
        logger.error("❌ IoT dataset loading test failed!")
        sys.exit(1)
