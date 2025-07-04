"""
Refactored models module implementing core interfaces.

This module provides production-grade model implementations following
SOLID principles and implementing the ModelInterface.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    # Fallback for environments without torch
    TORCH_AVAILABLE = False
    nn = None

from ..core.abstractions import BaseModel
from ..core.interfaces import ModelInterface
from ..core.types import ModelParameters, Metrics, ClientConfig
from ..core.exceptions import ModelError, ConfigurationError

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron model implementing ModelInterface.
    
    Features:
    - Configurable architecture via config
    - Proper initialization and parameter handling
    - Built-in training and evaluation logic
    - Error handling and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP model.
        
        Args:
            config: Model configuration containing:
                - input_dim: Input dimension
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - activation: Activation function (default: 'relu')
                - dropout_rate: Dropout rate (default: 0.0)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ModelError("PyTorch is required for MLP model")
        
        # Validate required config
        required_keys = ['input_dim', 'output_dim']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required MLP config key: {key}")
        
        self.input_dim = self.config['input_dim']
        self.hidden_dims = self.config.get('hidden_dims', [128, 64])
        self.output_dim = self.config['output_dim']
        self.activation = self.config.get('activation', 'relu')
        self.dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Build the model
        self.model = self._build_model()
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.optimizer_type = self.config.get('optimizer', 'adam')
        self.loss_function = self.config.get('loss_function', 'cross_entropy')
        
        # Initialize optimizer and loss
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        
        self.logger.info(f"Initialized MLP model: {self.input_dim} -> {self.hidden_dims} -> {self.output_dim}")
    
    def _build_model(self) -> nn.Module:
        """Build the MLP architecture."""
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add activation
            if self.activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif self.activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ConfigurationError(f"Unsupported activation: {self.activation}")
            
            # Add dropout if specified
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type.lower() == 'adamw':
            weight_decay = self.config.get('weight_decay', 0.01)
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            raise ConfigurationError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.loss_function.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_function.lower() == 'mse':
            return nn.MSELoss()
        elif self.loss_function.lower() == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ConfigurationError(f"Unsupported loss function: {self.loss_function}")
    
    def get_parameters(self) -> ModelParameters:
        """
        Get model parameters as a list of numpy arrays.
        
        Returns:
            List of parameter arrays
        """
        try:
            return [param.detach().cpu().numpy() for param in self.model.parameters()]
        except Exception as e:
            self.logger.error(f"Failed to get model parameters: {e}")
            raise ModelError(f"Could not extract model parameters: {e}") from e
    
    def set_parameters(self, parameters: ModelParameters) -> None:
        """
        Set model parameters from a list of numpy arrays.
        
        Args:
            parameters: List of parameter arrays
            
        Raises:
            ModelError: If parameter setting fails
        """
        try:
            if not isinstance(parameters, list):
                raise ModelError("Parameters must be a list of arrays")
            
            model_params = list(self.model.parameters())
            if len(parameters) != len(model_params):
                raise ModelError(
                    f"Parameter count mismatch: expected {len(model_params)}, got {len(parameters)}"
                )
            
            with torch.no_grad():
                for model_param, new_param in zip(model_params, parameters):
                    # Convert numpy to tensor if needed
                    if hasattr(new_param, 'shape'):  # numpy array
                        param_tensor = torch.from_numpy(new_param)
                    else:
                        param_tensor = torch.tensor(new_param)
                    
                    # Ensure shapes match
                    if model_param.shape != param_tensor.shape:
                        raise ModelError(
                            f"Parameter shape mismatch: expected {model_param.shape}, "
                            f"got {param_tensor.shape}"
                        )
                    
                    model_param.copy_(param_tensor)
            
            self.logger.debug("Model parameters updated successfully")
            
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"Failed to set model parameters: {e}")
            raise ModelError(f"Could not set model parameters: {e}") from e
    
    def train(self, data: Any) -> Metrics:
        """
        Train the model on provided data.
        
        Args:
            data: Training data (DataLoader or tuple of (features, targets))
            
        Returns:
            Training metrics
        """
        try:
            self.model.train()
            
            # Handle different data types
            if isinstance(data, DataLoader):
                return self._train_with_dataloader(data)
            elif isinstance(data, tuple) and len(data) == 2:
                return self._train_with_tensors(data[0], data[1])
            else:
                raise ModelError("Data must be a DataLoader or tuple of (features, targets)")
                
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"Training failed: {e}")
            raise ModelError(f"Model training failed: {e}") from e
    
    def _train_with_dataloader(self, dataloader: DataLoader) -> Metrics:
        """Train with DataLoader."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy for classification
            if self.loss_function.lower() == 'cross_entropy':
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
            
            total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def _train_with_tensors(self, features: torch.Tensor, targets: torch.Tensor) -> Metrics:
        """Train with tensor data."""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy for classification
        accuracy = 0.0
        if self.loss_function.lower() == 'cross_entropy':
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(targets)
        }
    
    def evaluate(self, data: Any) -> Metrics:
        """
        Evaluate the model on provided data.
        
        Args:
            data: Evaluation data (DataLoader or tuple of (features, targets))
            
        Returns:
            Evaluation metrics
        """
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Handle different data types
                if isinstance(data, DataLoader):
                    return self._evaluate_with_dataloader(data)
                elif isinstance(data, tuple) and len(data) == 2:
                    return self._evaluate_with_tensors(data[0], data[1])
                else:
                    raise ModelError("Data must be a DataLoader or tuple of (features, targets)")
                    
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"Evaluation failed: {e}")
            raise ModelError(f"Model evaluation failed: {e}") from e
    
    def _evaluate_with_dataloader(self, dataloader: DataLoader) -> Metrics:
        """Evaluate with DataLoader."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in dataloader:
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            
            # Calculate accuracy for classification
            if self.loss_function.lower() == 'cross_entropy':
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
            
            total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def _evaluate_with_tensors(self, features: torch.Tensor, targets: torch.Tensor) -> Metrics:
        """Evaluate with tensor data."""
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy for classification
        accuracy = 0.0
        if self.loss_function.lower() == 'cross_entropy':
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(targets)
        }


class LSTMModel(BaseModel):
    """
    LSTM model implementing ModelInterface.
    
    Features:
    - Configurable LSTM architecture
    - Sequence processing capabilities
    - Built-in training and evaluation logic
    - Proper parameter handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM model.
        
        Args:
            config: Model configuration containing:
                - input_dim: Input dimension
                - hidden_dim: Hidden dimension
                - num_layers: Number of LSTM layers
                - output_dim: Output dimension
                - dropout_rate: Dropout rate (default: 0.0)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ModelError("PyTorch is required for LSTM model")
        
        # Validate required config
        required_keys = ['input_dim', 'hidden_dim', 'num_layers', 'output_dim']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required LSTM config key: {key}")
        
        self.input_dim = self.config['input_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.num_layers = self.config['num_layers']
        self.output_dim = self.config['output_dim']
        self.dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Build the model
        self.model = self._build_model()
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.optimizer_type = self.config.get('optimizer', 'adam')
        self.loss_function = self.config.get('loss_function', 'cross_entropy')
        
        # Initialize optimizer and loss
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        
        self.logger.info(
            f"Initialized LSTM model: {self.input_dim} -> "
            f"{self.hidden_dim}x{self.num_layers} -> {self.output_dim}"
        )
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM architecture."""
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
                super(LSTMNet, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_dim, 
                    hidden_dim, 
                    num_layers, 
                    batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_dim, output_dim)
                
                if dropout_rate > 0:
                    self.dropout = nn.Dropout(dropout_rate)
                else:
                    self.dropout = None
            
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                batch_size = x.size(0)
                device = x.device
                
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                
                # LSTM forward pass
                out, _ = self.lstm(x, (h0, c0))
                
                # Take the output of the last time step
                out = out[:, -1, :]
                
                # Apply dropout if configured
                if self.dropout is not None:
                    out = self.dropout(out)
                
                # Final linear layer
                out = self.fc(out)
                return out
        
        return LSTMNet(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            self.output_dim, 
            self.dropout_rate
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer (same as MLP)."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type.lower() == 'adamw':
            weight_decay = self.config.get('weight_decay', 0.01)
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            raise ConfigurationError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function (same as MLP)."""
        if self.loss_function.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_function.lower() == 'mse':
            return nn.MSELoss()
        elif self.loss_function.lower() == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ConfigurationError(f"Unsupported loss function: {self.loss_function}")
    
    def get_parameters(self) -> ModelParameters:
        """Get model parameters (same implementation as MLP)."""
        try:
            return [param.detach().cpu().numpy() for param in self.model.parameters()]
        except Exception as e:
            self.logger.error(f"Failed to get model parameters: {e}")
            raise ModelError(f"Could not extract model parameters: {e}") from e
    
    def set_parameters(self, parameters: ModelParameters) -> None:
        """Set model parameters (same implementation as MLP)."""
        try:
            if not isinstance(parameters, list):
                raise ModelError("Parameters must be a list of arrays")
            
            model_params = list(self.model.parameters())
            if len(parameters) != len(model_params):
                raise ModelError(
                    f"Parameter count mismatch: expected {len(model_params)}, got {len(parameters)}"
                )
            
            with torch.no_grad():
                for model_param, new_param in zip(model_params, parameters):
                    if hasattr(new_param, 'shape'):  # numpy array
                        param_tensor = torch.from_numpy(new_param)
                    else:
                        param_tensor = torch.tensor(new_param)
                    
                    if model_param.shape != param_tensor.shape:
                        raise ModelError(
                            f"Parameter shape mismatch: expected {model_param.shape}, "
                            f"got {param_tensor.shape}"
                        )
                    
                    model_param.copy_(param_tensor)
            
            self.logger.debug("LSTM model parameters updated successfully")
            
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"Failed to set model parameters: {e}")
            raise ModelError(f"Could not set model parameters: {e}") from e
    
    def train(self, data: Any) -> Metrics:
        """Train the LSTM model (same implementation as MLP)."""
        try:
            self.model.train()
            
            if isinstance(data, DataLoader):
                return self._train_with_dataloader(data)
            elif isinstance(data, tuple) and len(data) == 2:
                return self._train_with_tensors(data[0], data[1])
            else:
                raise ModelError("Data must be a DataLoader or tuple of (features, targets)")
                
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"LSTM training failed: {e}")
            raise ModelError(f"LSTM model training failed: {e}") from e
    
    def _train_with_dataloader(self, dataloader: DataLoader) -> Metrics:
        """Train with DataLoader (same as MLP)."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in dataloader:
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.loss_function.lower() == 'cross_entropy':
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
            
            total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def _train_with_tensors(self, features: torch.Tensor, targets: torch.Tensor) -> Metrics:
        """Train with tensor data (same as MLP)."""
        self.optimizer.zero_grad()
        
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        accuracy = 0.0
        if self.loss_function.lower() == 'cross_entropy':
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(targets)
        }
    
    def evaluate(self, data: Any) -> Metrics:
        """Evaluate the LSTM model (same implementation as MLP)."""
        try:
            self.model.eval()
            
            with torch.no_grad():
                if isinstance(data, DataLoader):
                    return self._evaluate_with_dataloader(data)
                elif isinstance(data, tuple) and len(data) == 2:
                    return self._evaluate_with_tensors(data[0], data[1])
                else:
                    raise ModelError("Data must be a DataLoader or tuple of (features, targets)")
                    
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            self.logger.error(f"LSTM evaluation failed: {e}")
            raise ModelError(f"LSTM model evaluation failed: {e}") from e
    
    def _evaluate_with_dataloader(self, dataloader: DataLoader) -> Metrics:
        """Evaluate with DataLoader (same as MLP)."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in dataloader:
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            
            if self.loss_function.lower() == 'cross_entropy':
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
            
            total_samples += batch_targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def _evaluate_with_tensors(self, features: torch.Tensor, targets: torch.Tensor) -> Metrics:
        """Evaluate with tensor data (same as MLP)."""
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        
        accuracy = 0.0
        if self.loss_function.lower() == 'cross_entropy':
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(targets)
        }


class ModelRegistry:
    """
    Registry for model classes following the Registry pattern.
    
    Allows easy extension and configuration-driven model selection.
    """
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class."""
        if not issubclass(model_class, ModelInterface):
            raise ConfigurationError(f"Model class must implement ModelInterface")
        
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get_model_class(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            raise ConfigurationError(f"Unknown model: {name}")
        
        return cls._models[name]
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> ModelInterface:
        """Create a model instance from configuration."""
        model_name = config.get('type')
        if not model_name:
            raise ConfigurationError("Model type not specified in config")
        
        model_class = cls.get_model_class(model_name)
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())


# Register built-in models
ModelRegistry.register('mlp', MLPModel)
ModelRegistry.register('lstm', LSTMModel)
