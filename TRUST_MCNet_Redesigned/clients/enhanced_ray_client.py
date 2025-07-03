"""
Enhanced Ray-based Flower client for TRUST-MCNet federated learning.

This module implements a Ray Actor that wraps the Flower NumPyClient interface,
enabling distributed client execution with resource management, trust mechanisms,
and improved error handling.
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
from torch.utils.data import DataLoader, Subset
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

from models.model import MLP, LSTM
from utils.data_utils import create_data_loaders, split_train_eval
from utils.ray_utils import cleanup_training_resources, cleanup_evaluation_resources
from trust_module.trust_evaluator import TrustEvaluator

logger = logging.getLogger(__name__)


@ray.remote
class EnhancedRayFlowerClient:
    """
    Enhanced Ray Actor implementing Flower NumPyClient interface.
    
    This client runs as a Ray actor with improved:
    - Memory management and cleanup
    - Error handling with retries
    - Multi-epoch local training support
    - Gradient-free evaluation
    - Resource monitoring
    """
    
    def __init__(
        self, 
        client_id: str,
        dataset_subset: Subset,
        cfg: Dict[str, Any]
    ):
        """
        Initialize enhanced Ray Flower client.
        
        Args:
            client_id: Unique identifier for this client
            dataset_subset: Client's data subset
            cfg: Complete configuration dictionary
        """
        self.client_id = client_id
        self.cfg = cfg
        self.device = self._setup_device()
        
        # Initialize model with error handling
        try:
            self.model = self._create_model()
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Client {client_id}: Failed to create model: {e}")
            raise
        
        # Split client data into train/eval with validation
        try:
            train_subset, eval_subset = split_train_eval(
                dataset_subset, 
                eval_fraction=cfg['dataset']['eval_fraction']
            )
            
            # Validate data splits
            if len(train_subset) == 0:
                raise ValueError(f"Client {client_id}: No training data after split")
            if len(eval_subset) == 0:
                logger.warning(f"Client {client_id}: No evaluation data after split")
            
        except Exception as e:
            logger.error(f"Client {client_id}: Failed to split data: {e}")
            raise
        
        # Create optimized data loaders
        try:
            dataloader_config = cfg.get('env', {}).get('dataloader', {})
            self.train_loader, self.eval_loader = create_data_loaders(
                train_subset,
                eval_subset,
                batch_size=cfg['dataset']['batch_size'],
                num_workers=dataloader_config.get('num_workers', 0)
            )
        except Exception as e:
            logger.error(f"Client {client_id}: Failed to create data loaders: {e}")
            raise
        
        # Initialize optimizer with error handling
        try:
            self.optimizer = self._create_optimizer()
            self.lr_scheduler = self._create_lr_scheduler()
        except Exception as e:
            logger.error(f"Client {client_id}: Failed to create optimizer: {e}")
            raise
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking
        self.performance_history = []
        self.training_metrics = {}
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"Enhanced client {client_id} initialized: "
                   f"{len(train_subset)} train, {len(eval_subset)} eval samples on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device with error handling."""
        try:
            device_config = self.cfg['env']['device']
            
            if device_config == 'auto':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    logger.debug(f"Client {self.client_id}: Using CUDA device")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device('mps')
                    logger.debug(f"Client {self.client_id}: Using MPS device")
                else:
                    device = torch.device('cpu')
                    logger.debug(f"Client {self.client_id}: Using CPU device")
            else:
                device = torch.device(device_config)
                logger.debug(f"Client {self.client_id}: Using configured device: {device_config}")
            
            return device
            
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Device setup failed, using CPU: {e}")
            return torch.device('cpu')
    
    def _create_model(self) -> nn.Module:
        """Create model with enhanced error handling."""
        model_config = self.cfg['model']
        
        try:
            if model_config['type'] == 'MLP':
                model = MLP(
                    input_dim=model_config['mlp']['input_dim'],
                    hidden_dims=model_config['mlp']['hidden_dims'],
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
            
            # Apply dropout if configured
            if hasattr(model, 'apply_dropout'):
                dropout_rate = self.cfg.get('training', {}).get('dropout', 0.0)
                if dropout_rate > 0:
                    model.apply_dropout(dropout_rate)
            
            return model
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Model creation failed: {e}")
            raise
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with enhanced configuration support."""
        training_config = self.cfg['training']
        optimizer_name = training_config['optimizer'].lower()
        lr = training_config['learning_rate']
        weight_decay = training_config['weight_decay']
        
        try:
            if optimizer_name == 'adam':
                betas = training_config.get('betas', [0.9, 0.999])
                return optim.Adam(
                    self.model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay,
                    betas=betas
                )
            elif optimizer_name == 'sgd':
                momentum = training_config.get('momentum', 0.9)
                nesterov = training_config.get('nesterov', False)
                return optim.SGD(
                    self.model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov
                )
            elif optimizer_name == 'rmsprop':
                return optim.RMSprop(
                    self.model.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Optimizer creation failed: {e}")
            raise
    
    def _create_lr_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler if configured."""
        training_config = self.cfg['training']
        scheduler_type = training_config.get('lr_scheduler')
        
        if scheduler_type is None:
            return None
        
        try:
            if scheduler_type == 'step':
                step_size = training_config.get('lr_step_size', 10)
                gamma = training_config.get('lr_decay', 0.1)
                return optim.lr_scheduler.StepLR(
                    self.optimizer, 
                    step_size=step_size, 
                    gamma=gamma
                )
            elif scheduler_type == 'exponential':
                gamma = training_config.get('lr_decay', 0.95)
                return optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 
                    gamma=gamma
                )
            elif scheduler_type == 'cosine':
                T_max = training_config.get('epochs', 1)
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=T_max
                )
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}")
                return None
                
        except Exception as e:
            logger.warning(f"Client {self.client_id}: LR scheduler creation failed: {e}")
            return None
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """Get model parameters with error handling."""
        try:
            parameters = []
            for param in self.model.parameters():
                param_array = param.detach().cpu().numpy()
                parameters.append(param_array)
            
            logger.debug(f"Client {self.client_id}: Retrieved {len(parameters)} parameter arrays")
            return parameters
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error getting parameters: {e}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters with validation."""
        try:
            model_params = list(self.model.parameters())
            
            if len(parameters) != len(model_params):
                raise ValueError(f"Parameter count mismatch: expected {len(model_params)}, "
                               f"got {len(parameters)}")
            
            for model_param, new_param in zip(model_params, parameters):
                if model_param.shape != new_param.shape:
                    raise ValueError(f"Parameter shape mismatch: expected {model_param.shape}, "
                                   f"got {new_param.shape}")
                
                model_param.data = torch.tensor(new_param, dtype=model_param.dtype).to(self.device)
            
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
        Train model with enhanced multi-epoch support and error handling.
        
        Args:
            parameters: Global model parameters
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Set global parameters
                self.set_parameters(parameters)
                
                # Extract training configuration
                epochs = config.get('epochs', self.cfg['training']['epochs'])
                gradient_clipping = self.cfg['training'].get('gradient_clipping', None)
                
                # Train model
                self.model.train()
                total_loss = 0.0
                total_examples = 0
                epoch_metrics = []
                
                for epoch in range(epochs):
                    epoch_start_time = time.time()
                    epoch_loss = 0.0
                    epoch_examples = 0
                    
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                        try:
                            data, target = data.to(self.device), target.to(self.device)
                            
                            # Flatten data for MLP models
                            if isinstance(self.model, MLP):
                                data = data.view(data.size(0), -1)
                            
                            # Forward pass
                            self.optimizer.zero_grad()
                            output = self.model(data)
                            loss = self.criterion(output, target)
                            
                            # Backward pass
                            loss.backward()
                            
                            # Gradient clipping if configured
                            if gradient_clipping is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    gradient_clipping
                                )
                            
                            self.optimizer.step()
                            
                            # Update metrics
                            epoch_loss += loss.item()
                            epoch_examples += len(data)
                            
                        except Exception as batch_error:
                            logger.warning(f"Client {self.client_id}: Batch {batch_idx} failed: {batch_error}")
                            continue
                    
                    # Update learning rate scheduler
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Calculate epoch metrics
                    if epoch_examples > 0:
                        avg_epoch_loss = epoch_loss / len(self.train_loader)
                        epoch_time = time.time() - epoch_start_time
                        
                        epoch_metrics.append({
                            'epoch': epoch + 1,
                            'loss': avg_epoch_loss,
                            'time': epoch_time,
                            'examples': epoch_examples
                        })
                        
                        total_loss += epoch_loss
                        total_examples = epoch_examples  # Use last epoch's count
                        
                        logger.debug(f"Client {self.client_id}: Epoch {epoch+1}/{epochs}, "
                                   f"Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
                
                # Cleanup resources after training
                cleanup_training_resources()
                
                # Calculate overall metrics
                training_time = time.time() - start_time
                avg_loss = total_loss / (epochs * len(self.train_loader)) if epochs > 0 else 0.0
                
                # Get updated parameters
                updated_parameters = self.get_parameters({})
                
                # Prepare metrics
                metrics = {
                    'train_loss': avg_loss,
                    'training_time': training_time,
                    'epochs_completed': epochs,
                    'total_examples': total_examples,
                    'client_id': self.client_id,
                    'epoch_metrics': epoch_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Store performance history
                self.performance_history.append({
                    'timestamp': time.time(),
                    'type': 'training',
                    'metrics': metrics
                })
                
                logger.info(f"Client {self.client_id}: Training completed in {training_time:.2f}s, "
                           f"Avg Loss: {avg_loss:.4f}, Examples: {total_examples}")
                
                return updated_parameters, total_examples, metrics
                
            except Exception as e:
                logger.error(f"Client {self.client_id}: Training attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Client {self.client_id}: Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"Client {self.client_id}: All training attempts failed")
                    # Return original parameters on final failure
                    return parameters, 0, {'error': str(e), 'client_id': self.client_id}
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model with gradient-free computation and error handling.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration from server
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Set global parameters
                self.set_parameters(parameters)
                
                # Evaluate model with no gradient computation
                self.model.eval()
                test_loss = 0.0
                correct = 0
                total_examples = 0
                
                with torch.no_grad():  # Disable gradient tracking for evaluation
                    for batch_idx, (data, target) in enumerate(self.eval_loader):
                        try:
                            data, target = data.to(self.device), target.to(self.device)
                            
                            # Flatten data for MLP models
                            if isinstance(self.model, MLP):
                                data = data.view(data.size(0), -1)
                            
                            # Forward pass
                            output = self.model(data)
                            loss = self.criterion(output, target)
                            
                            # Update metrics
                            test_loss += loss.item()
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                            total_examples += len(data)
                            
                        except Exception as batch_error:
                            logger.warning(f"Client {self.client_id}: Eval batch {batch_idx} failed: {batch_error}")
                            continue
                
                # Cleanup resources after evaluation
                cleanup_evaluation_resources()
                
                # Calculate metrics
                evaluation_time = time.time() - start_time
                avg_loss = test_loss / len(self.eval_loader) if len(self.eval_loader) > 0 else 0.0
                accuracy = correct / total_examples if total_examples > 0 else 0.0
                
                metrics = {
                    'accuracy': accuracy,
                    'correct_predictions': correct,
                    'evaluation_time': evaluation_time,
                    'client_id': self.client_id
                }
                
                # Store performance history
                self.performance_history.append({
                    'timestamp': time.time(),
                    'type': 'evaluation',
                    'metrics': metrics
                })
                
                logger.info(f"Client {self.client_id}: Evaluation completed in {evaluation_time:.2f}s, "
                           f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                return avg_loss, total_examples, metrics
                
            except Exception as e:
                logger.error(f"Client {self.client_id}: Evaluation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Client {self.client_id}: Retrying evaluation in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"Client {self.client_id}: All evaluation attempts failed")
                    return float('inf'), 0, {'error': str(e), 'client_id': self.client_id}
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get client performance history."""
        return self.performance_history
    
    def get_trust_metrics(self) -> Dict[str, Any]:
        """Get trust-related metrics for this client."""
        try:
            # Calculate trust metrics based on performance history
            if not self.performance_history:
                return {'trust_score': 0.5, 'client_id': self.client_id}
            
            recent_performance = self.performance_history[-5:]  # Last 5 rounds
            
            # Calculate average accuracy and loss stability
            accuracies = [p['metrics'].get('accuracy', 0.0) for p in recent_performance 
                         if p['type'] == 'evaluation']
            losses = [p['metrics'].get('train_loss', float('inf')) for p in recent_performance 
                     if p['type'] == 'training']
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            loss_stability = 1.0 / (1.0 + np.std(losses)) if losses else 0.5
            
            trust_score = 0.7 * avg_accuracy + 0.3 * loss_stability
            
            return {
                'trust_score': float(trust_score),
                'avg_accuracy': float(avg_accuracy),
                'loss_stability': float(loss_stability),
                'performance_samples': len(recent_performance),
                'client_id': self.client_id
            }
            
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Failed to calculate trust metrics: {e}")
            return {'trust_score': 0.5, 'client_id': self.client_id}


def create_ray_client_fn(client_subsets: List[Subset], cfg: Dict[str, Any]):
    """
    Create Ray-based client function for Flower simulation.
    
    Args:
        client_subsets: List of dataset subsets for each client
        cfg: Configuration dictionary
        
    Returns:
        Client function for Flower simulation
    """
    def client_fn(cid: str):
        """Create client instance."""
        try:
            client_id = int(cid)
            if client_id >= len(client_subsets):
                raise ValueError(f"Client ID {client_id} out of range [0, {len(client_subsets)})")
            
            dataset_subset = client_subsets[client_id]
            
            # Create enhanced Ray client
            ray_client = EnhancedRayFlowerClient.remote(
                client_id=f"client_{client_id}",
                dataset_subset=dataset_subset,
                cfg=cfg
            )
            
            return RayClientWrapper(ray_client)
            
        except Exception as e:
            logger.error(f"Failed to create client {cid}: {e}")
            raise
    
    return client_fn


class RayClientWrapper(fl.client.NumPyClient):
    """Wrapper to make Ray actor compatible with Flower NumPyClient interface."""
    
    def __init__(self, ray_client):
        """Initialize wrapper with Ray client actor."""
        self.ray_client = ray_client
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """Get parameters from Ray client."""
        return ray.get(self.ray_client.get_parameters.remote(config))
    
    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict[str, Any]]:
        """Train model using Ray client."""
        return ray.get(self.ray_client.fit.remote(parameters, config))
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model using Ray client."""
        return ray.get(self.ray_client.evaluate.remote(parameters, config))
