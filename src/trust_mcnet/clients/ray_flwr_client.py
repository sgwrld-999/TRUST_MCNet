"""
Ray-based Flower client for TRUST-MCNet federated learning.

This module implements a Ray Actor that wraps the Flower NumPyClient interface,
enabling distributed client execution with resource management, trust mechanisms,
and improved training logic.
"""

import logging
import time
import traceback
import gc
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import ray
import flwr as fl
from flwr.common import (
    NDArrays, 
    Parameters, 
    FitIns, 
    FitRes, 
    EvaluateIns, 
    EvaluateRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from torch.utils.data import DataLoader, Subset

from models.model import MLP, LSTM
from utils.data_utils import create_data_loaders, split_train_eval
from trust_module.trust_evaluator import TrustEvaluator
from utils.ray_utils import cleanup_training_resources, cleanup_evaluation_resources, MemoryTracker

logger = logging.getLogger(__name__)


@ray.remote
class RayFlowerClient:
    """
    Ray Actor implementing Flower NumPyClient interface.
    
    This client runs as a Ray actor, enabling distributed execution
    and resource management for federated learning with trust mechanisms.
    """
    
    def __init__(
        self, 
        client_id: str,
        dataset_subset: Subset,
        cfg: Dict[str, Any]
    ):
        """
        Initialize Ray Flower client.
        
        Args:
            client_id: Unique identifier for this client
            dataset_subset: Client's data subset
            cfg: Complete configuration dictionary
        """
        self.client_id = client_id
        self.cfg = cfg
        self.device = self._setup_device()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Split client data into train/eval
        train_subset, eval_subset = split_train_eval(
            dataset_subset, 
            eval_fraction=cfg['dataset']['eval_fraction']
        )
        
        # Create data loaders
        self.train_loader, self.eval_loader = create_data_loaders(
            train_subset,
            eval_subset,
            batch_size=cfg['dataset']['batch_size'],
            num_workers=0  # Use 0 for Ray actors to avoid conflicts
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Initialized client {client_id} with {len(train_subset)} train, "
                   f"{len(eval_subset)} eval samples on device {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device based on configuration."""
        device_config = self.cfg['env']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        return device
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.cfg['model']
        
        if model_config['type'] == 'MLP':
            model = MLP(
                input_dim=model_config['mlp']['input_dim'],
                output_dim=model_config['mlp']['output_dim']
            )
        elif model_config['type'] == 'LSTM':
            model = LSTM(
                input_dim=model_config['lstm']['input_dim'],
                hidden_dim=model_config['lstm']['hidden_dim'],
                num_layers=model_config['lstm']['num_layers'],
                output_dim=model_config['lstm']['output_dim']
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.cfg['training']['optimizer'].lower()
        lr = self.cfg['training']['learning_rate']
        weight_decay = self.cfg['training']['weight_decay']
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """
        Get model parameters as NumPy arrays.
        
        Args:
            config: Configuration from server
            
        Returns:
            Model parameters as list of NumPy arrays
        """
        try:
            # Convert model parameters to numpy arrays
            parameters = []
            for param in self.model.parameters():
                parameters.append(param.detach().cpu().numpy())
            
            logger.debug(f"Client {self.client_id}: Retrieved {len(parameters)} parameter arrays")
            return parameters
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error getting parameters: {e}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from NumPy arrays.
        
        Args:
            parameters: Model parameters as list of NumPy arrays
        """
        try:
            # Load parameters into model
            params_dict = zip(self.model.parameters(), parameters)
            for model_param, new_param in params_dict:
                model_param.data = torch.tensor(new_param).to(self.device)
            
            logger.debug(f"Client {self.client_id}: Set {len(parameters)} parameter arrays")
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error setting parameters: {e}")
            raise
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[NDArrays, int, Dict[str, Any]]:
        """
        Train model on client's data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            start_time = time.time()
            
            # Set global parameters
            self.set_parameters(parameters)
            
            # Train model
            self.model.train()
            train_loss = 0.0
            num_examples = 0
            
            epochs = config.get('epochs', self.cfg['training']['epochs'])
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_examples = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Flatten data for MLP
                    if isinstance(self.model, MLP):
                        data = data.view(data.size(0), -1)
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_examples += len(data)
                
                train_loss += epoch_loss
                num_examples = epoch_examples  # Use last epoch's count
                
                logger.debug(f"Client {self.client_id}: Epoch {epoch+1}/{epochs}, "
                           f"Loss: {epoch_loss/len(self.train_loader):.4f}")
            
            # Calculate average loss
            avg_loss = train_loss / (epochs * len(self.train_loader))
            
            # Get updated parameters
            updated_parameters = self.get_parameters({})
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Metrics to return
            metrics = {
                'train_loss': avg_loss,
                'training_time': training_time,
                'epochs_completed': epochs,
                'client_id': self.client_id
            }
            
            logger.info(f"Client {self.client_id}: Training completed in {training_time:.2f}s, "
                       f"Loss: {avg_loss:.4f}, Examples: {num_examples}")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during training: {e}")
            logger.error(traceback.format_exc())
            
            # Return original parameters on error
            return parameters, 0, {'error': str(e)}
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on client's test data.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration from server
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            # Set global parameters
            self.set_parameters(parameters)
            
            # Evaluate model
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in self.eval_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Flatten data for MLP
                    if isinstance(self.model, MLP):
                        data = data.view(data.size(0), -1)
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    test_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            # Calculate metrics
            avg_loss = test_loss / len(self.eval_loader)
            accuracy = correct / total if total > 0 else 0.0
            
            # Store performance for trust evaluation
            performance_metrics = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'correct': correct,
                'total': total
            }
            self.performance_history.append(performance_metrics)
            
            # Metrics to return
            metrics = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'client_id': self.client_id
            }
            
            logger.info(f"Client {self.client_id}: Evaluation completed, "
                       f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return avg_loss, total, metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during evaluation: {e}")
            logger.error(traceback.format_exc())
            
            return float('inf'), 0, {'error': str(e)}
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get client information and statistics.
        
        Returns:
            Dictionary containing client information
        """
        try:
            info = {
                'client_id': self.client_id,
                'device': str(self.device),
                'model_type': self.cfg['model']['type'],
                'train_samples': len(self.train_loader.dataset),
                'eval_samples': len(self.eval_loader.dataset),
                'performance_history': self.performance_history,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error getting client info: {e}")
            return {'client_id': self.client_id, 'error': str(e)}


class FlowerClientWrapper(fl.client.NumPyClient):
    """
    Wrapper to adapt RayFlowerClient to Flower's NumPyClient interface.
    
    This wrapper allows Ray actors to be used seamlessly with Flower's
    simulation framework.
    """
    
    def __init__(self, ray_client_ref):
        """
        Initialize wrapper with Ray client reference.
        
        Args:
            ray_client_ref: Reference to Ray actor client
        """
        self.ray_client = ray_client_ref
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """Get parameters from Ray client."""
        return ray.get(self.ray_client.get_parameters.remote(config))
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[NDArrays, int, Dict[str, Any]]:
        """Fit model using Ray client."""
        return ray.get(self.ray_client.fit.remote(parameters, config))
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model using Ray client."""
        return ray.get(self.ray_client.evaluate.remote(parameters, config))


def create_ray_client_fn(client_subsets: List[Subset], cfg: Dict[str, Any]):
    """
    Create a client function for Flower simulation that returns Ray-based clients.
    
    Args:
        client_subsets: List of dataset subsets for each client
        cfg: Configuration dictionary
        
    Returns:
        Client function compatible with Flower simulation
    """
    def client_fn(cid: str) -> FlowerClientWrapper:
        """
        Create a client for the given client ID.
        
        Args:
            cid: Client ID as string
            
        Returns:
            FlowerClientWrapper wrapping a Ray actor client
        """
        try:
            # Parse client ID to integer index
            client_idx = int(cid)
            
            if client_idx >= len(client_subsets):
                raise ValueError(f"Client ID {client_idx} out of range")
            
            # Get client's dataset subset
            client_subset = client_subsets[client_idx]
            
            # Create Ray actor for this client
            ray_client = RayFlowerClient.remote(
                client_id=cid,
                dataset_subset=client_subset,
                cfg=cfg
            )
            
            # Return wrapped client
            return FlowerClientWrapper(ray_client)
            
        except Exception as e:
            logger.error(f"Error creating client {cid}: {e}")
            raise
    
    return client_fn
