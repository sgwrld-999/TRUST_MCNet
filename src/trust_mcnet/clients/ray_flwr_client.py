"""
Ray-based Flower client for TRUST-MCNet federated learning.

This module implements Ray Actors that wrap the Flower NumPyClient interface,
enabling distributed client execution with resource management, trust mechanisms,
and improved training logic. The module combines functionality from both
enhanced_ray_client.py and ray_flwr_client.py into a unified implementation.

Key features:
- Dynamic resource detection and allocation
- Memory tracking and optimization
- Advanced error handling and recovery
- Comprehensive metrics collection
- Trust evaluation integration
- Model fingerprinting for security
"""

import logging
import time
import traceback
import gc
import os
import psutil
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


@ray.remote(num_cpus=1, num_gpus=0.2)
class RayFlowerClient:
    """
    Enhanced Ray Actor implementing Flower NumPyClient interface.
    
    This client runs as a Ray actor, enabling distributed execution
    and resource management for federated learning with advanced trust mechanisms.
    
    Key features:
    - Automatic resource detection and allocation
    - Memory optimization and tracking
    - Error handling with graceful recovery
    - Trust evaluation integration
    - Comprehensive metrics collection
    - Model fingerprinting for security
    Features:
    - Automatic resource cleanup and memory tracking
    - Adaptive learning rate scheduling
    - Fault tolerance with graceful error handling
    - Performance history tracking
    - Model fingerprinting for trust verification
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
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(client_id=client_id, log_interval=10)
        self.memory_tracker.start()
        
        try:
            # Initialize model with error handling
            self.model = self._create_model()
            self.model.to(self.device)
            
            # Split client data into train/eval with error handling
            train_subset, eval_subset = self._prepare_data_split(dataset_subset)
            
            # Create data loaders with optimized settings
            self.train_loader, self.eval_loader = self._create_data_loaders(train_subset, eval_subset)
            
            # Enhanced optimization setup
            self._setup_training_components()
            
            # Performance tracking with extended metrics
            self.performance_history = []
            self.training_metrics = {
                'round_times': [],
                'loss_history': [],
                'accuracy_history': []
            }
            
            # Model fingerprinting for trust verification
            self.model_fingerprint = self._compute_model_fingerprint()
            
            logger.info(f"Initialized client {client_id} with {len(train_subset)} train, "
                      f"{len(eval_subset)} eval samples on device {self.device}")
                      
        except Exception as e:
            logger.error(f"Client {client_id} initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            # Attempt resource cleanup
            self._cleanup_resources()
            raise
    
    def _prepare_data_split(self, dataset_subset: Subset) -> Tuple[Subset, Subset]:
        """
        Prepare train/eval data split with error handling.
        
        Args:
            dataset_subset: Client's data subset
            
        Returns:
            Tuple of (train_subset, eval_subset)
            
        Raises:
            ValueError: If dataset subset is invalid or empty
        """
        if dataset_subset is None or len(dataset_subset) == 0:
            raise ValueError(f"Client {self.client_id}: Empty dataset provided")
            
        try:
            # Get split configuration
            eval_fraction = self.cfg['dataset'].get('eval_fraction', 0.2)
            # Ensure we have at least one sample in each split
            min_samples_required = max(2, int(1 / eval_fraction))
            
            if len(dataset_subset) < min_samples_required:
                logger.warning(
                    f"Client {self.client_id}: Dataset too small ({len(dataset_subset)} samples) "
                    f"for requested split. Using 50/50 split instead of {eval_fraction}."
                )
                eval_fraction = 0.5
            
            # Split the data
            train_subset, eval_subset = split_train_eval(dataset_subset, eval_fraction)
            
            # Verify split
            if len(train_subset) == 0 or len(eval_subset) == 0:
                raise ValueError(f"Invalid data split: train={len(train_subset)}, eval={len(eval_subset)}")
                
            return train_subset, eval_subset
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Data split failed: {str(e)}")
            raise ValueError(f"Data preparation failed: {str(e)}")
    
    def _create_data_loaders(self, train_subset: Subset, eval_subset: Subset) -> Tuple[DataLoader, DataLoader]:
        """
        Create optimized data loaders with performance considerations.
        
        Args:
            train_subset: Training data subset
            eval_subset: Evaluation data subset
            
        Returns:
            Tuple of (train_loader, eval_loader)
        """
        batch_size = self.cfg['dataset'].get('batch_size', 32)
        
        # Optimize batch size based on dataset size
        if len(train_subset) < batch_size * 2:
            adjusted_batch_size = max(1, len(train_subset) // 2)
            logger.warning(
                f"Client {self.client_id}: Dataset too small for batch size {batch_size}. "
                f"Adjusted to {adjusted_batch_size}."
            )
            batch_size = adjusted_batch_size
        
        # Pin memory if using CUDA
        pin_memory = self.device.type == 'cuda'
        
        # Create loaders
        return create_data_loaders(
            train_subset,
            eval_subset,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for Ray actors to avoid conflicts
            pin_memory=pin_memory
        )
        
    def _setup_training_components(self) -> None:
        """
        Set up enhanced training components including optimizer,
        scheduler, loss function and regularization.
        """
        # Initialize optimizer with configuration
        self.optimizer = self._create_optimizer()
        
        # Add learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Enhanced loss function
        self._setup_loss_function()
        
        # Setup regularization if configured
        self._setup_regularization()
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.cfg.get('training', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', None)
        
        if not scheduler_type:
            return None
            
        if scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            return ExponentialLR(
                self.optimizer, 
                gamma=scheduler_config.get('gamma', 0.9)
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 10),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None
            
    def _setup_loss_function(self) -> None:
        """Set up appropriate loss function based on configuration."""
        loss_type = self.cfg.get('training', {}).get('loss', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            logger.warning(f"Unknown loss type: {loss_type}. Using CrossEntropyLoss.")
            self.criterion = nn.CrossEntropyLoss()
            
    def _setup_regularization(self) -> None:
        """Set up regularization techniques based on configuration."""
        reg_config = self.cfg.get('training', {}).get('regularization', {})
        
        # Weight decay (L2 regularization) is typically set in optimizer
        self.use_l1_reg = reg_config.get('l1', 0.0) > 0
        self.l1_lambda = reg_config.get('l1', 0.0)
        
        # Dropout is typically built into model architecture
        
        # Early stopping handled at server level
        
    def _compute_model_fingerprint(self) -> Dict[str, Any]:
        """
        Compute a model fingerprint for trust verification.
        
        Returns:
            Dictionary containing model signature information
        """
        fingerprint = {
            'architecture': type(self.model).__name__,
            'param_count': sum(p.numel() for p in self.model.parameters()),
            'layer_shapes': {},
            'gradient_norms': {}
        }
        
        # Add parameter shapes
        for name, param in self.model.named_parameters():
            fingerprint['layer_shapes'][name] = list(param.shape)
            if param.grad is not None:
                fingerprint['gradient_norms'][name] = float(param.grad.norm().item())
        
        return fingerprint
        
    def _cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            logger.info(f"Client {self.client_id}: Cleaning up resources")
            
            # Log final memory stats
            if hasattr(self, 'memory_tracker'):
                self.memory_tracker.log_stats()
            
            # Clear model and optimizer
            if hasattr(self, 'model'):
                # Delete model parameters explicitly
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                self.model = None
                
            if hasattr(self, 'optimizer'):
                self.optimizer = None
            
            if hasattr(self, 'scheduler'):
                self.scheduler = None
                
            # Close data loaders
            if hasattr(self, 'train_loader'):
                del self.train_loader
            if hasattr(self, 'eval_loader'):
                del self.eval_loader
                
            # Clear other large objects
            if hasattr(self, 'performance_history'):
                self.performance_history.clear()
                
            if hasattr(self, 'training_metrics'):
                self.training_metrics.clear()
                
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Client {self.client_id}: Resources cleaned up successfully")
                
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Error during resource cleanup: {str(e)}")
            
    def cleanup(self) -> Dict[str, Any]:
        """
        Public method to clean up client resources and return final stats.
        
        Returns:
            Dictionary with final client statistics
        """
        # Get final statistics
        final_stats = {
            'client_id': self.client_id,
            'device_used': str(self.device),
            'training_completed': len(self.performance_history),
            'final_memory': self.memory_tracker.get_stats() if hasattr(self, 'memory_tracker') else None
        }
        
        # Add final performance if available
        if hasattr(self, 'performance_history') and self.performance_history:
            final_stats['final_performance'] = self.performance_history[-1]
            
        # Perform cleanup
        self._cleanup_resources()
        
        return final_stats
    
    def _setup_device(self) -> torch.device:
        """
        Set up computing device with proper error handling and fallback.
        
        Returns:
            Configured torch device (CPU or CUDA)
        """
        device_config = self.cfg['env'].get('device', 'auto')
        device = torch.device('cpu')  # Default fallback
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                try:
                    # Check available CUDA memory
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    # Convert to MB
                    free_memory_mb = free_memory / (1024 * 1024)
                    
                    if free_memory_mb > 200:  # Require at least 200MB free
                        device = torch.device('cuda')
                        logger.info(f"Client {self.client_id} using CUDA with {free_memory_mb:.0f}MB free")
                    else:
                        logger.warning(f"CUDA available but only {free_memory_mb:.0f}MB free. Falling back to CPU.")
                except Exception as e:
                    logger.warning(f"Error setting up CUDA: {str(e)}. Falling back to CPU.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info(f"Client {self.client_id} using MPS (Apple Silicon)")
            else:
                logger.info(f"Client {self.client_id} using CPU")
        else:
            try:
                device = torch.device(device_config)
                logger.info(f"Client {self.client_id} using configured device: {device_config}")
            except Exception as e:
                logger.warning(f"Invalid device '{device_config}', falling back to CPU: {str(e)}")
        
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
        Train model on client's data with enhanced monitoring and resilience.
        
        Args:
            parameters: Global model parameters
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            start_time = time.time()
            round_num = config.get('current_round', 0)
            
            # Track memory before training
            mem_before = self.memory_tracker.sample()
            
            # Set global parameters
            self.set_parameters(parameters)
            
            # Get training configuration
            epochs = config.get('epochs', self.cfg['training']['epochs'])
            lr_factor = config.get('lr_factor', 1.0)  # Server can dynamically adjust learning rate
            
            # Apply learning rate adjustment if provided
            if lr_factor != 1.0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_factor
                    current_lr = param_group['lr']
                    logger.info(f"Client {self.client_id}: Adjusted learning rate to {current_lr:.6f}")
            
            # Initialize training metrics
            train_loss = 0.0
            num_examples = 0
            correct_predictions = 0
            total_examples = 0
            batch_times = []
            
            # Train model with enhanced monitoring
            self.model.train()
            
            # Track per-epoch metrics
            epoch_metrics = []
            
            for epoch in range(epochs):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                epoch_examples = 0
                epoch_correct = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    batch_start = time.time()
                    
                    # Move data to device with error handling
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                    except Exception as e:
                        logger.error(f"Client {self.client_id}: Error moving data to device: {e}")
                        # Try to continue with CPU as fallback
                        self.device = torch.device('cpu')
                        self.model.to(self.device)
                        data, target = data.to(self.device), target.to(self.device)
                    
                    # Reshape data based on model type
                    if isinstance(self.model, MLP):
                        data = data.view(data.size(0), -1)
                    
                    # Forward-backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Add L1 regularization if configured
                    if hasattr(self, 'use_l1_reg') and self.use_l1_reg:
                        l1_penalty = torch.tensor(0.0).to(self.device)
                        for param in self.model.parameters():
                            l1_penalty += torch.sum(torch.abs(param))
                        loss += self.l1_lambda * l1_penalty
                    
                    loss.backward()
                    
                    # Gradient clipping for stability
                    if self.cfg.get('training', {}).get('clip_gradients', False):
                        max_norm = self.cfg.get('training', {}).get('max_norm', 1.0)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    
                    self.optimizer.step()
                    
                    # Calculate batch metrics
                    batch_loss = loss.item()
                    batch_size = len(data)
                    
                    # Track accuracy
                    _, predicted = torch.max(output.data, 1)
                    batch_correct = (predicted == target).sum().item()
                    
                    # Update counters
                    epoch_loss += batch_loss
                    epoch_examples += batch_size
                    epoch_correct += batch_correct
                    
                    # Track batch time
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    # Sample memory periodically
                    if batch_idx % 10 == 0:
                        self.memory_tracker.sample()
                
                # Calculate epoch metrics
                epoch_accuracy = epoch_correct / epoch_examples if epoch_examples > 0 else 0
                epoch_avg_loss = epoch_loss / len(self.train_loader) if self.train_loader else 0
                epoch_time = time.time() - epoch_start_time
                
                # Store epoch metrics
                epoch_metrics.append({
                    'epoch': epoch + 1,
                    'loss': epoch_avg_loss,
                    'accuracy': epoch_accuracy,
                    'time': epoch_time
                })
                
                # Update running totals
                train_loss += epoch_loss
                num_examples = epoch_examples  # Use last epoch's count
                correct_predictions += epoch_correct
                total_examples += epoch_examples
                
                # Log epoch results
                logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{epochs}, "
                          f"Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                          f"Time: {epoch_time:.2f}s")
                
                # Apply scheduler if configured
                if hasattr(self, 'scheduler') and self.scheduler:
                    self.scheduler.step()
            
            # Calculate final metrics
            training_time = time.time() - start_time
            avg_loss = train_loss / (epochs * len(self.train_loader)) if self.train_loader and epochs > 0 else 0
            accuracy = correct_predictions / total_examples if total_examples > 0 else 0
            
            # Get memory usage after training
            mem_after = self.memory_tracker.sample()
            mem_diff = mem_after - mem_before
            
            # Compute model fingerprint for trust verification
            model_fingerprint = self._compute_model_fingerprint()
            
            # Get updated parameters
            updated_parameters = self.get_parameters({})
            
            # Store training metrics for history
            self.training_metrics['round_times'].append(training_time)
            self.training_metrics['loss_history'].append(avg_loss)
            self.training_metrics['accuracy_history'].append(accuracy)
            
            # Enhanced metrics
            metrics = {
                'train_loss': avg_loss,
                'train_accuracy': accuracy,
                'training_time': training_time,
                'epochs_completed': epochs,
                'client_id': self.client_id,
                'memory_usage_mb': {
                    'before': mem_before,
                    'after': mem_after,
                    'diff': mem_diff,
                    'peak': self.memory_tracker.peak_memory
                },
                'batch_time_stats': {
                    'mean': np.mean(batch_times),
                    'min': np.min(batch_times),
                    'max': np.max(batch_times),
                    'std': np.std(batch_times)
                },
                'epoch_metrics': epoch_metrics,
                'model_fingerprint': model_fingerprint,
                'round_number': round_num
            }
            
            # Run garbage collection to free memory
            gc.collect()
            
            logger.info(f"Client {self.client_id}: Training completed in {training_time:.2f}s, "
                       f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                       f"Memory change: {mem_diff:.2f}MB, Examples: {num_examples}")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during training: {e}")
            logger.error(traceback.format_exc())
            
            # Free memory on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory stats on error
            self.memory_tracker.log_stats()
            
            # Return original parameters on error with diagnostic information
            return parameters, 0, {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'client_id': self.client_id,
                'memory_stats': self.memory_tracker.get_stats()
            }
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on client's test data with enhanced metrics and trust verification.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration from server
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            start_time = time.time()
            round_num = config.get('current_round', 0)
            
            # Track memory before evaluation
            mem_before = self.memory_tracker.sample()
            
            # Set global parameters
            self.set_parameters(parameters)
            
            # Prepare for evaluation
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            # Initialize prediction tracking
            all_predictions = []
            all_targets = []
            class_correct = {}
            class_total = {}
            
            # Run evaluation with detailed metrics collection
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.eval_loader):
                    try:
                        # Move data to device with error handling
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Reshape data based on model type
                        if isinstance(self.model, MLP):
                            data = data.view(data.size(0), -1)
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        test_loss += loss.item()
                        
                        # Calculate accuracy
                        _, predicted = torch.max(output.data, 1)
                        batch_size = target.size(0)
                        total += batch_size
                        batch_correct = (predicted == target).sum().item()
                        correct += batch_correct
                        
                        # Store predictions and targets for detailed analysis
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                        
                        # Track per-class accuracy
                        for i in range(batch_size):
                            label = target[i].item()
                            pred = predicted[i].item()
                            
                            if label not in class_total:
                                class_total[label] = 0
                                class_correct[label] = 0
                                
                            class_total[label] += 1
                            if label == pred:
                                class_correct[label] += 1
                                
                        # Sample memory periodically
                        if batch_idx % 10 == 0:
                            self.memory_tracker.sample()
                            
                    except Exception as e:
                        logger.error(f"Client {self.client_id}: Error in evaluation batch {batch_idx}: {e}")
                        # Continue with next batch
            
            # Calculate metrics
            avg_loss = test_loss / len(self.eval_loader) if self.eval_loader else 0
            accuracy = correct / total if total > 0 else 0.0
            eval_time = time.time() - start_time
            
            # Calculate per-class accuracy
            class_metrics = {}
            for label in class_total:
                class_acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
                class_metrics[f"class_{label}_accuracy"] = class_acc
                class_metrics[f"class_{label}_samples"] = class_total[label]
            
            # Get memory usage after evaluation
            mem_after = self.memory_tracker.sample()
            mem_diff = mem_after - mem_before
            
            # Calculate confusion matrix if available
            confusion_matrix = None
            if len(all_predictions) > 0:
                try:
                    import numpy as np
                    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                    unique_labels = sorted(set(all_targets))
                    confusion_matrix = sk_confusion_matrix(
                        all_targets, all_predictions, labels=unique_labels
                    ).tolist()
                except ImportError:
                    logger.warning("sklearn not available for confusion matrix calculation")
            
            # Store performance metrics for history and trust evaluation
            performance_metrics = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'correct': correct,
                'total': total,
                'time': eval_time,
                'round': round_num,
                'class_metrics': class_metrics,
                'memory_usage': {
                    'before': mem_before,
                    'after': mem_after,
                    'diff': mem_diff
                }
            }
            self.performance_history.append(performance_metrics)
            
            # Enhanced metrics to return
            metrics = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'client_id': self.client_id,
                'evaluation_time': eval_time,
                'memory_usage_mb': {
                    'before': mem_before,
                    'after': mem_after,
                    'diff': mem_diff
                },
                'round_number': round_num,
                'class_metrics': class_metrics
            }
            
            # Add optional metrics if available
            if confusion_matrix:
                metrics['confusion_matrix'] = confusion_matrix
            
            # Run garbage collection to free memory
            gc.collect()
            
            logger.info(f"Client {self.client_id}: Evaluation completed in {eval_time:.2f}s, "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Examples: {total}")
            
            return avg_loss, total, metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error during evaluation: {e}")
            logger.error(traceback.format_exc())
            
            # Free memory on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory stats on error
            self.memory_tracker.log_stats()
            
            # Return infinity on error with diagnostic information
            return float('inf'), 0, {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'client_id': self.client_id,
                'memory_stats': self.memory_tracker.get_stats()
            }
    
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
