#!/usr/bin/env python3
"""
Simple client simulator for testing the trust-weighted Flower server.

This script creates a basic Flower client that can connect to the trust-weighted
server for testing purposes.

Usage:
    python client/simulate.py --cid 0
    python client/simulate.py --cid 1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import flwr as fl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


class SimpleTestClient(fl.client.NumPyClient):
    """Simple test client for trust-weighted server validation."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model_params = self._initialize_params()
        
    def _initialize_params(self) -> list:
        """Initialize simple model parameters."""
        # Simple 2-layer MLP for MNIST-like data (784 -> 128 -> 10)
        np.random.seed(int(self.client_id))
        
        params = [
            np.random.normal(0, 0.1, (784, 128)).astype(np.float32),  # W1
            np.zeros(128, dtype=np.float32),                          # b1
            np.random.normal(0, 0.1, (128, 10)).astype(np.float32),  # W2
            np.zeros(10, dtype=np.float32)                            # b2
        ]
        
        return params
    
    def get_parameters(self, config: Dict[str, Any]) -> list:
        """Return current model parameters."""
        return self.model_params.copy()
    
    def fit(
        self, 
        parameters: list, 
        config: Dict[str, Any]
    ) -> Tuple[list, int, Dict[str, Any]]:
        """Simulate local training."""
        # Update local parameters
        self.model_params = [p.copy() for p in parameters]
        
        # Simulate training by adding small noise
        noise_scale = 0.01
        for i, param in enumerate(self.model_params):
            noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
            self.model_params[i] += noise
        
        # Simulate training metrics
        num_samples = np.random.randint(50, 200)
        accuracy = np.random.uniform(0.7, 0.95)
        loss = np.random.uniform(0.1, 0.5)
        
        # Add client-specific bias for trust evaluation testing
        client_bias = int(self.client_id) * 0.05
        accuracy = max(0.0, min(1.0, accuracy - client_bias))
        loss = max(0.0, loss + client_bias)
        
        metrics = {
            'accuracy': float(accuracy),
            'train_loss': float(loss),
            'client_id': self.client_id,
            'epochs_completed': 3
        }
        
        logger.info(f"Client {self.client_id}: Training completed. "
                   f"Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")
        
        return self.model_params, num_samples, metrics
    
    def evaluate(
        self, 
        parameters: list, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Simulate evaluation."""
        # Update parameters for evaluation
        self.model_params = [p.copy() for p in parameters]
        
        # Simulate evaluation metrics
        num_samples = np.random.randint(20, 50)
        accuracy = np.random.uniform(0.6, 0.9)
        loss = np.random.uniform(0.2, 0.6)
        
        metrics = {
            'accuracy': float(accuracy),
            'client_id': self.client_id
        }
        
        logger.info(f"Client {self.client_id}: Evaluation completed. "
                   f"Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")
        
        return float(loss), num_samples, metrics


def main():
    """Main client simulator entry point."""
    parser = argparse.ArgumentParser(description='TRUST_MCNet Test Client')
    parser.add_argument('--cid', type=str, required=True, help='Client ID')
    parser.add_argument('--server', type=str, default='localhost:8080', 
                       help='Server address')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=f'[Client {args.cid}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting client {args.cid}")
    logger.info(f"Connecting to server: {args.server}")
    
    try:
        # Create client
        client = SimpleTestClient(args.cid)
        
        # Start client
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client
        )
        
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)
    finally:
        logger.info("Client shutdown complete")


if __name__ == "__main__":
    main()
