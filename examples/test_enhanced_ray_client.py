"""
Test script to verify the enhanced RayFlowerClient implementation.

This script creates a simple test case to verify the functionality of the enhanced
RayFlowerClient, including memory tracking, error handling, and metrics collection.
"""

import os
import sys
import logging
import time
import traceback
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import ray
from torch.utils.data import TensorDataset

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print(f"Added path: {project_root}")

try:
    from src.trust_mcnet.clients.ray_flwr_client import RayFlowerClient
except ImportError:
    print("Could not import from src.trust_mcnet, trying trust_mcnet...")
    from trust_mcnet.clients.ray_flwr_client import RayFlowerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=10):
    """Create a synthetic dataset for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)

def test_ray_flower_client():
    """Test the enhanced RayFlowerClient implementation."""
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True)
            
        logger.info("Creating synthetic dataset...")
        dataset = create_synthetic_dataset()
        
        # Configure client
        client_config = {
            'model': {
                'type': 'mlp',
                'input_dim': 20,
                'hidden_dims': [64, 32],
                'output_dim': 10,
                'activation': 'relu',
                'dropout': 0.2
            },
            'training': {
                'batch_size': 32,
                'epochs': 2,
                'optimizer': {
                    'type': 'adam',
                    'lr': 0.001,
                    'weight_decay': 0.0001
                },
                'scheduler': {
                    'type': 'step_lr',
                    'step_size': 5,
                    'gamma': 0.9
                },
                'loss': 'cross_entropy',
                'clip_gradients': True,
                'max_norm': 1.0,
                'regularization': {
                    'l1': 0.0001
                }
            },
            'dataset': {
                'eval_fraction': 0.2
            }
        }
        
        # Create client actor
        logger.info("Creating client actor...")
        client_ref = RayFlowerClient.remote(
            client_id="test_client_001",
            dataset_subset=dataset,
            cfg=client_config
        )
        
        # Create dummy model parameters
        logger.info("Creating dummy model parameters...")
        dummy_model = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10)
        )
        dummy_parameters = [p.cpu().numpy() for p in dummy_model.parameters()]
        
        # Test fit method
        logger.info("Testing fit method...")
        fit_config = {
            'epochs': 3,
            'current_round': 1,
            'lr_factor': 0.95
        }
        fit_result = ray.get(client_ref.fit.remote(dummy_parameters, fit_config))
        parameters, num_examples, metrics = fit_result
        
        logger.info(f"Fit completed with {num_examples} examples")
        logger.info(f"Metrics: {metrics}")
        
        # Test evaluate method
        logger.info("Testing evaluate method...")
        evaluate_config = {
            'current_round': 1
        }
        eval_result = ray.get(client_ref.evaluate.remote(parameters, evaluate_config))
        loss, num_examples, metrics = eval_result
        
        logger.info(f"Evaluation completed with {num_examples} examples")
        logger.info(f"Loss: {loss}, Metrics: {metrics}")
        
        # Test cleanup method
        logger.info("Testing cleanup method...")
        cleanup_result = ray.get(client_ref.cleanup.remote())
        
        logger.info(f"Cleanup completed with result: {cleanup_result}")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    test_ray_flower_client()
