"""
Enhanced neural network models for TRUST-MCNet federated learning framework.

This module provides improved implementations of MLP and LSTM models with:
- Comprehensive documentation and type hints
- Input validation and error handling  
- Configurable architectures
- Proper weight initialization
- Support for different normalization strategies
"""

import logging
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Raised when model configuration or input validation fails."""
    pass


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all TRUST-MCNet models.
    
    Provides common functionality and ensures consistent interface
    across different model architectures.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self._validate_dimensions(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def _validate_dimensions(self, input_dim: int, output_dim: int) -> None:
        """Validate input and output dimensions."""
        if input_dim <= 0:
            raise ModelValidationError(f"Input dimension must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ModelValidationError(f"Output dimension must be positive, got {output_dim}")
    
    @abstractmethod
    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part of the network."""
        pass
    
    def get_model_size(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedMLP(BaseModel):
    """
    Enhanced Multi-Layer Perceptron for IoT anomaly detection in federated learning.
    
    Features:
    - Configurable architecture with sensible defaults
    - Multiple normalization options (LayerNorm, BatchNorm)  
    - Dropout regularization
    - Proper weight initialization
    - Comprehensive input validation
    
    Default Architecture:
        Input -> FC(1024) -> ReLU -> FC(512) -> ReLU -> LayerNorm -> 
        FC(256) -> ReLU -> FC(128) -> ReLU -> LayerNorm -> 
        FC(64) -> ReLU -> FC(32) -> ReLU -> FC(16) -> ReLU -> Output
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes  
        hidden_dims: Custom hidden layer dimensions (optional)
        dropout_rate: Dropout probability for regularization (0.0 = no dropout)
        use_batch_norm: Use BatchNorm instead of LayerNorm
        activation: Activation function ('relu', 'leaky_relu', 'gelu')
        
    Example:
        >>> model = EnhancedMLP(input_dim=784, output_dim=10)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    SUPPORTED_ACTIVATIONS = {'relu', 'leaky_relu', 'gelu', 'elu'}
    DEFAULT_HIDDEN_DIMS = [1024, 512, 256, 128, 64, 32, 16]
    NORMALIZATION_POSITIONS = {1, 3}  # After 2nd and 4th hidden layers
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super().__init__(input_dim, output_dim)
        
        # Validate and set configuration
        self._validate_config(dropout_rate, activation)
        self.hidden_dims = hidden_dims or self.DEFAULT_HIDDEN_DIMS.copy()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        
        # Build network architecture
        self.layers = self._build_layers()
        self.activation_fn = self._get_activation_function(activation)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created EnhancedMLP: {self.get_model_size():,} parameters")
    
    def _validate_config(self, dropout_rate: float, activation: str) -> None:
        """Validate model configuration parameters."""
        if not (0.0 <= dropout_rate < 1.0):
            raise ModelValidationError(f"Dropout rate must be in [0, 1), got {dropout_rate}")
        
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ModelValidationError(
                f"Activation must be one of {self.SUPPORTED_ACTIVATIONS}, got '{activation}'"
            )
    
    def _build_layers(self) -> nn.ModuleList:
        """
        Build the network layers with proper normalization and regularization.
        
        Returns:
            ModuleList containing all network layers
        """
        layers = nn.ModuleList()
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Skip activation and normalization for output layer
            if i == len(layer_dims) - 2:
                break
                
            # Activation function (handled in forward pass)
            
            # Normalization at specific positions
            if i in self.NORMALIZATION_POSITIONS:
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                else:
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
            
            # Dropout for regularization
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        return layers
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function based on name."""
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'elu': nn.ELU()
        }
        return activation_map[activation]
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for ReLU-like activations
                if self.activation_name in ['relu', 'leaky_relu']:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                # Initialize bias to small positive value
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
            
        Raises:
            ModelValidationError: If input tensor has wrong shape
        """
        self._validate_input(x)
        
        layer_idx = 0
        
        # Process layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                
                # Apply activation (except for output layer)
                if layer_idx < len(self.hidden_dims):
                    x = self.activation_fn(x)
                    
                layer_idx += 1
            else:
                # Normalization or dropout layer
                x = layer(x)
        
        return x
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if x.dim() != 2:
            raise ModelValidationError(f"Expected 2D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_dim:
            raise ModelValidationError(
                f"Expected input size {self.input_dim}, got {x.size(1)}"
            )
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction part of the network (without final layer).
        
        Returns:
            Sequential module containing all layers except the output layer
        """
        feature_layers = []
        layer_idx = 0
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if layer_idx == len(self.hidden_dims):  # Skip output layer
                    break
                layer_idx += 1
            feature_layers.append(layer)
        
        return nn.Sequential(*feature_layers)
    
    def get_architecture_summary(self) -> str:
        """Get a human-readable summary of the architecture."""
        summary_lines = [
            f"EnhancedMLP Architecture:",
            f"  Input: {self.input_dim}",
        ]
        
        for i, dim in enumerate(self.hidden_dims):
            summary_lines.append(f"  Hidden {i+1}: {dim} ({self.activation_name})")
            if i in self.NORMALIZATION_POSITIONS:
                norm_type = "BatchNorm" if self.use_batch_norm else "LayerNorm"
                summary_lines.append(f"    -> {norm_type}")
            if self.dropout_rate > 0:
                summary_lines.append(f"    -> Dropout({self.dropout_rate})")
        
        summary_lines.append(f"  Output: {self.output_dim}")
        summary_lines.append(f"  Total parameters: {self.get_model_size():,}")
        
        return "\n".join(summary_lines)


class EnhancedLSTM(BaseModel):
    """
    Enhanced LSTM model for sequential IoT data processing.
    
    Features:
    - Bidirectional LSTM option
    - Configurable hidden dimensions and layers
    - Dropout regularization  
    - Proper weight initialization
    - Attention mechanism option
    
    Args:
        input_dim: Number of input features per timestep
        output_dim: Number of output classes
        hidden_dim: Hidden state dimension
        num_layers: Number of LSTM layers
        dropout_rate: Dropout probability
        bidirectional: Use bidirectional LSTM
        use_attention: Add attention mechanism
        
    Example:
        >>> model = EnhancedLSTM(input_dim=10, output_dim=2, hidden_dim=64)
        >>> x = torch.randn(32, 50, 10)  # (batch, seq_len, features)
        >>> output = model(x)  # Shape: (32, 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        super().__init__(input_dim, output_dim)
        
        # Validate configuration
        if hidden_dim <= 0:
            raise ModelValidationError(f"Hidden dimension must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ModelValidationError(f"Number of layers must be positive, got {num_layers}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ModelValidationError(f"Dropout rate must be in [0, 1), got {dropout_rate}")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Calculate actual hidden size (bidirectional doubles it)
        lstm_hidden_size = hidden_dim * (2 if bidirectional else 1)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden_size,
                num_heads=min(8, lstm_hidden_size // 8),
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Output layer
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.output_layer = nn.Linear(lstm_hidden_size, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created EnhancedLSTM: {self.get_model_size():,} parameters")
    
    def _initialize_weights(self) -> None:
        """Initialize LSTM and linear layer weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (standard practice)
                n = param.size(0)
                param[n//4:n//2].fill_(1.0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        self._validate_input(x)
        
        batch_size = x.size(0)
        
        # Initialize hidden states
        h_0, c_0 = self._init_hidden_states(batch_size, x.device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over sequence
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use mean pooling over sequence
            sequence_repr = attn_out.mean(dim=1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                sequence_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                sequence_repr = h_n[-1]
        
        # Apply dropout and output layer
        output = self.dropout(sequence_repr)
        output = self.output_layer(output)
        
        return output
    
    def _init_hidden_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states."""
        num_directions = 2 if self.bidirectional else 1
        
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        
        return h_0, c_0
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if x.dim() != 3:
            raise ModelValidationError(f"Expected 3D input tensor, got {x.dim()}D")
        
        if x.size(2) != self.input_dim:
            raise ModelValidationError(
                f"Expected input feature size {self.input_dim}, got {x.size(2)}"
            )
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the LSTM feature extraction part (without output layer).
        
        Returns:
            LSTM module for feature extraction
        """
        class LSTMFeatureExtractor(nn.Module):
            def __init__(self, lstm_model):
                super().__init__()
                self.lstm = lstm_model.lstm
                self.attention = getattr(lstm_model, 'attention', None)
                self.use_attention = lstm_model.use_attention
                self.bidirectional = lstm_model.bidirectional
                self.hidden_dim = lstm_model.hidden_dim
                self.num_layers = lstm_model.num_layers
            
            def forward(self, x):
                batch_size = x.size(0)
                device = x.device
                
                # Initialize hidden states
                num_directions = 2 if self.bidirectional else 1
                h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
                c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
                
                lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
                
                if self.use_attention and self.attention is not None:
                    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    return attn_out.mean(dim=1)
                else:
                    if self.bidirectional:
                        return torch.cat([h_n[-2], h_n[-1]], dim=1)
                    else:
                        return h_n[-1]
        
        return LSTMFeatureExtractor(self)


# Legacy compatibility aliases
MLP = EnhancedMLP
LSTM = EnhancedLSTM


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp' or 'lstm')
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific parameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return EnhancedMLP(input_dim, output_dim, **kwargs)
    elif model_type == 'lstm':
        return EnhancedLSTM(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'mlp' or 'lstm'.")
