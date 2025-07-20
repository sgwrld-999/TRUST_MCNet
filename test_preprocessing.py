#!/usr/bin/env python3
"""
Test script to validate the adaptive feature selection preprocessing.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trust_mcnet.utils.dataset_registry import IoTGeneralLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing():
    """Test the adaptive feature selection preprocessing."""
    
    # Sample IoT-like data with redundant features
    data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00', '2024-01-01 10:03:00'],
        'unique_id': ['id_001', 'id_002', 'id_003', 'id_004'],
        'orig_host': ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4'],
        'resp_host': ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4'],
        'orig_port': [443, 80, 22, 8080],
        'resp_port': [80, 443, 22, 8080],
        'duration': [1.5, 2.1, 0.8, 3.2],
        'orig_bytes': [1024, 2048, 512, 4096],
        'resp_bytes': [512, 1024, 256, 2048],
        'protocol': ['TCP', 'TCP', 'SSH', 'HTTP'],
        'service': ['https', 'http', 'ssh', 'http-proxy'],
        'constant_col': [1, 1, 1, 1],  # Should be removed
        'high_cardinality': ['val1', 'val2', 'val3', 'val4'],  # Should be removed (100% unique)
        'Label': ['Benign', 'Malicious', 'Benign', 'Malicious']
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created test dataframe with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Config similar to IoT general config
    config = {
        'label_config': {
            'target_column': 'Label'
        },
        'preprocessing': {
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
                'max_features': 10
            }
        }
    }
    
    # Test preprocessing
    loader = IoTGeneralLoader()
    try:
        processed_df = loader._preprocess_dataframe(df, config)
        logger.info(f"Processed dataframe shape: {processed_df.shape}")
        logger.info(f"Remaining columns: {list(processed_df.columns)}")
        
        # Expected: timestamp, unique_id, orig_host, resp_host, constant_col, high_cardinality should be removed
        removed_expected = ['timestamp', 'unique_id', 'orig_host', 'resp_host', 'constant_col', 'high_cardinality']
        remaining_expected = ['orig_port', 'resp_port', 'duration', 'orig_bytes', 'resp_bytes', 'protocol', 'service', 'Label']
        
        removed_actual = [col for col in df.columns if col not in processed_df.columns]
        logger.info(f"Actually removed: {removed_actual}")
        logger.info(f"Expected to remove: {removed_expected}")
        
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing adaptive feature selection preprocessing...")
    success = test_preprocessing()
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
        sys.exit(1)
