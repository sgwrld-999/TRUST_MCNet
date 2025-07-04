"""
Refactored federated learning clients using interface-based architecture.

This module implements clients that conform to our new interfaces,
providing:
- Interface-based design for extensibility
- Registry pattern for client selection
- Production-grade error handling
- Comprehensive resource management
- Scalable client execution
"""

import logging
import time
import gc
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC
import numpy as np

# Import framework-specific modules when available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import flwr as fl
    from flwr.common import (
        NDArrays, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
        Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

from core.interfaces import ClientInterface, ModelInterface, DataLoaderInterface, TrustEvaluatorInterface
from core.abstractions import BaseClient
from core.exceptions import ClientError, ModelError, DataLoadingError, TrustEvaluationError
from core.types import ConfigType, ModelWeights, TrainingResults, EvaluationResults

logger = logging.getLogger(__name__)


class FederatedLearningClient(BaseClient):
    """
    Standard federated learning client implementation.
    
    Implements the client-side federated learning logic with:
    - Model training and evaluation
    - Trust evaluation integration
    - Resource management
    - Error handling and recovery
    """
    
    def __init__(
        self,
        client_id: str,
        config: ConfigType,
        model: ModelInterface,
        data_loader: DataLoaderInterface,
        trust_evaluator: Optional[TrustEvaluatorInterface] = None,
        **kwargs
    ) -> None:
        """
        Initialize federated learning client.
        
        Args:
            client_id: Unique client identifier
            config: Client configuration
            model: Model interface implementation
            data_loader: Data loader interface implementation
            trust_evaluator: Optional trust evaluator
            **kwargs: Additional parameters
        """
        super().__init__(client_id, config, **kwargs)
        
        self.model = model
        self.data_loader = data_loader
        self.trust_evaluator = trust_evaluator
        
        # Training configuration
        self.local_epochs = config.get('local_epochs', 1)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.optimizer_type = config.get('optimizer', 'sgd')
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        self.trust_scores = []
        
        # Resource tracking
        self.resource_usage = {
            'training_time': 0.0,
            'evaluation_time': 0.0,
            'memory_peak': 0.0,
            'data_size': 0
        }
        
        logger.info(f"Initialized FL client {client_id}")
    
    def fit(
        self,
        parameters: ModelWeights,
        config: Dict[str, Any],
        **kwargs
    ) -> TrainingResults:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            **kwargs: Additional parameters
            
        Returns:
            Training results including updated weights and metrics
            
        Raises:
            ClientError: If training fails
        """
        try:
            start_time = time.time()
            logger.debug(f"Client {self.client_id} starting fit")
            
            # Update local model with global parameters
            self.model.set_weights(parameters)
            
            # Get local training data
            train_loader = self.data_loader.get_train_loader(
                batch_size=config.get('batch_size', self.batch_size)
            )
            
            if len(train_loader.dataset) == 0:
                raise ClientError(f"Client {self.client_id} has no training data")
            
            # Train the model
            training_metrics = self._train_local_model(
                train_loader,
                epochs=config.get('epochs', self.local_epochs),
                learning_rate=config.get('learning_rate', self.learning_rate)
            )
            
            # Get updated model weights
            updated_weights = self.model.get_weights()
            
            # Calculate training time
            training_time = time.time() - start_time
            self.resource_usage['training_time'] += training_time
            
            # Prepare results
            results = {
                'weights': updated_weights,
                'num_samples': len(train_loader.dataset),
                'metrics': training_metrics,
                'client_id': self.client_id,
                'training_time': training_time
            }
            
            # Add trust evaluation if available
            if self.trust_evaluator:
                try:
                    trust_score = self.trust_evaluator.evaluate_trust(
                        client_id=self.client_id,
                        client_update=updated_weights,
                        round_num=config.get('round', 0)
                    )
                    results['trust_score'] = trust_score
                    self.trust_scores.append(trust_score)
                except Exception as e:
                    logger.warning(f"Trust evaluation failed for client {self.client_id}: {e}")
            
            # Store training history
            self.training_history.append(results.copy())
            
            logger.debug(f"Client {self.client_id} completed fit in {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in fit for client {self.client_id}: {e}")
            raise ClientError(f"Training failed: {e}") from e
        finally:
            # Cleanup
            self._cleanup_resources()
    
    def evaluate(
        self,
        parameters: ModelWeights,
        config: Dict[str, Any],
        **kwargs
    ) -> EvaluationResults:
        """
        Evaluate the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration
            **kwargs: Additional parameters
            
        Returns:
            Evaluation results including metrics
            
        Raises:
            ClientError: If evaluation fails
        """
        try:
            start_time = time.time()
            logger.debug(f"Client {self.client_id} starting evaluation")
            
            # Update local model with global parameters
            self.model.set_weights(parameters)
            
            # Get local test data
            test_loader = self.data_loader.get_test_loader(
                batch_size=config.get('batch_size', self.batch_size)
            )
            
            if len(test_loader.dataset) == 0:
                logger.warning(f"Client {self.client_id} has no test data")
                return {
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'num_samples': 0,
                    'client_id': self.client_id,
                    'evaluation_time': 0.0
                }
            
            # Evaluate the model
            evaluation_metrics = self._evaluate_local_model(test_loader)
            
            # Calculate evaluation time
            evaluation_time = time.time() - start_time
            self.resource_usage['evaluation_time'] += evaluation_time
            
            # Prepare results
            results = {
                'loss': evaluation_metrics.get('loss', float('inf')),
                'accuracy': evaluation_metrics.get('accuracy', 0.0),
                'num_samples': len(test_loader.dataset),
                'client_id': self.client_id,
                'evaluation_time': evaluation_time,
                'metrics': evaluation_metrics
            }
            
            # Store evaluation history
            self.evaluation_history.append(results.copy())
            
            logger.debug(f"Client {self.client_id} completed evaluation in {evaluation_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation for client {self.client_id}: {e}")
            raise ClientError(f"Evaluation failed: {e}") from e
        finally:
            # Cleanup
            self._cleanup_resources()
    
    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get client properties and capabilities.
        
        Args:
            config: Configuration
            
        Returns:
            Client properties
        """
        try:
            data_info = self.data_loader.get_data_info()
            
            properties = {
                'client_id': self.client_id,
                'data_size': data_info.get('total_samples', 0),
                'data_shape': data_info.get('data_shape'),
                'num_classes': data_info.get('num_classes'),
                'model_type': self.model.get_model_info().get('type'),
                'capabilities': ['fit', 'evaluate'],
                'resource_usage': self.resource_usage.copy(),
                'trust_enabled': self.trust_evaluator is not None
            }
            
            if self.training_history:
                properties['avg_training_time'] = np.mean([h['training_time'] for h in self.training_history])
            
            if self.evaluation_history:
                properties['avg_evaluation_time'] = np.mean([h['evaluation_time'] for h in self.evaluation_history])
            
            if self.trust_scores:
                properties['avg_trust_score'] = np.mean(self.trust_scores)
                properties['trust_trend'] = self.trust_scores[-1] - self.trust_scores[0] if len(self.trust_scores) > 1 else 0
            
            return properties
            
        except Exception as e:
            logger.error(f"Error getting properties for client {self.client_id}: {e}")
            raise ClientError(f"Failed to get properties: {e}") from e
    
    def _train_local_model(
        self,
        train_loader: DataLoader,
        epochs: int,
        learning_rate: float
    ) -> Dict[str, float]:
        """
        Train the local model for specified epochs.
        
        Args:
            train_loader: Training data loader
            epochs: Number of local epochs
            learning_rate: Learning rate
            
        Returns:
            Training metrics
        """
        try:
            # Set model to training mode
            self.model.train()
            
            # Setup optimizer
            optimizer = self._create_optimizer(learning_rate)
            
            total_loss = 0.0
            total_samples = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_samples = 0
                
                for batch_data in train_loader:
                    # Forward pass
                    loss = self.model.compute_loss(batch_data)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    batch_size = len(batch_data[0]) if isinstance(batch_data, (list, tuple)) else len(batch_data)
                    epoch_loss += loss.item() * batch_size
                    epoch_samples += batch_size
                
                total_loss += epoch_loss
                total_samples += epoch_samples
                
                logger.debug(f"Client {self.client_id} epoch {epoch + 1}/{epochs}, loss: {epoch_loss / epoch_samples:.4f}")
            
            avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
            
            return {
                'loss': avg_loss,
                'epochs': epochs,
                'samples_processed': total_samples
            }
            
        except Exception as e:
            logger.error(f"Error in local training for client {self.client_id}: {e}")
            raise ClientError(f"Local training failed: {e}") from e
    
    def _evaluate_local_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the local model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad() if TORCH_AVAILABLE else None:
                for batch_data in test_loader:
                    # Get predictions and loss
                    loss = self.model.compute_loss(batch_data)
                    predictions = self.model.predict(batch_data)
                    
                    # Calculate accuracy
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                        targets = batch_data[1]
                        if hasattr(predictions, 'argmax'):
                            predicted_classes = predictions.argmax(dim=1)
                        else:
                            predicted_classes = np.argmax(predictions, axis=1)
                        
                        if hasattr(targets, 'cpu'):
                            targets = targets.cpu()
                        if hasattr(predicted_classes, 'cpu'):
                            predicted_classes = predicted_classes.cpu()
                        
                        batch_correct = (predicted_classes == targets).sum().item() if hasattr((predicted_classes == targets), 'sum') else np.sum(predicted_classes == targets)
                        total_correct += batch_correct
                    
                    # Track metrics
                    batch_size = len(batch_data[0]) if isinstance(batch_data, (list, tuple)) else len(batch_data)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
            
            avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'total_samples': total_samples
            }
            
        except Exception as e:
            logger.error(f"Error in local evaluation for client {self.client_id}: {e}")
            raise ClientError(f"Local evaluation failed: {e}") from e
    
    def _create_optimizer(self, learning_rate: float):
        """Create optimizer for local training."""
        if not TORCH_AVAILABLE:
            raise ClientError("PyTorch not available for optimizer creation")
        
        model_params = self.model.get_parameters()
        
        if self.optimizer_type.lower() == 'sgd':
            return optim.SGD(model_params, lr=learning_rate)
        elif self.optimizer_type.lower() == 'adam':
            return optim.Adam(model_params, lr=learning_rate)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(model_params, lr=learning_rate)
        else:
            logger.warning(f"Unknown optimizer type {self.optimizer_type}, using SGD")
            return optim.SGD(model_params, lr=learning_rate)
    
    def _cleanup_resources(self) -> None:
        """Clean up resources after training/evaluation."""
        try:
            if TORCH_AVAILABLE:
                # Clear PyTorch cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")


# Ray-based client for distributed execution
if RAY_AVAILABLE:
    @ray.remote
    class RayFederatedClient(FederatedLearningClient):
        """
        Ray-based federated learning client for distributed execution.
        
        Extends the base client with Ray actor capabilities for
        scalable distributed federated learning.
        """
        
        def __init__(self, *args, **kwargs):
            """Initialize Ray client."""
            super().__init__(*args, **kwargs)
            logger.info(f"Initialized Ray FL client {self.client_id}")
        
        def get_actor_info(self) -> Dict[str, Any]:
            """Get Ray actor information."""
            return {
                'actor_id': ray.get_runtime_context().current_actor.actor_id.hex(),
                'node_id': ray.get_runtime_context().node_id.hex(),
                'client_id': self.client_id
            }


# Client Registry
class ClientRegistry:
    """Registry for federated learning clients."""
    
    _clients = {
        'standard': FederatedLearningClient,
    }
    
    if RAY_AVAILABLE:
        _clients['ray'] = RayFederatedClient
    
    @classmethod
    def register(cls, name: str, client_class: type) -> None:
        """Register a new client type."""
        if not issubclass(client_class, ClientInterface):
            raise ValueError(f"Client {client_class} must implement ClientInterface")
        cls._clients[name] = client_class
        logger.info(f"Registered client type: {name}")
    
    @classmethod
    def create(
        cls,
        name: str,
        client_id: str,
        config: ConfigType,
        model: ModelInterface,
        data_loader: DataLoaderInterface,
        trust_evaluator: Optional[TrustEvaluatorInterface] = None,
        **kwargs
    ) -> ClientInterface:
        """Create a client instance."""
        if name not in cls._clients:
            available = ', '.join(cls._clients.keys())
            raise ClientError(f"Client type '{name}' not found. Available: {available}")
        
        client_class = cls._clients[name]
        return client_class(client_id, config, model, data_loader, trust_evaluator, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available client types."""
        return list(cls._clients.keys())


def create_client(
    client_type: str,
    client_id: str,
    config: ConfigType,
    model: ModelInterface,
    data_loader: DataLoaderInterface,
    trust_evaluator: Optional[TrustEvaluatorInterface] = None,
    **kwargs
) -> ClientInterface:
    """
    Factory function to create client instances.
    
    Args:
        client_type: Type of client to create
        client_id: Unique client identifier
        config: Client configuration
        model: Model interface implementation
        data_loader: Data loader interface implementation
        trust_evaluator: Optional trust evaluator
        **kwargs: Additional parameters
        
    Returns:
        Client instance
    """
    return ClientRegistry.create(
        client_type, client_id, config, model, data_loader, trust_evaluator, **kwargs
    )


# Export public interface
__all__ = [
    'FederatedLearningClient',
    'ClientRegistry',
    'create_client'
]

if RAY_AVAILABLE:
    __all__.append('RayFederatedClient')
