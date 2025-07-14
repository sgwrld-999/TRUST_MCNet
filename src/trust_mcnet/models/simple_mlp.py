"""
Simple MLP model for TRUST_MCNet federated learning.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for federated learning.
    
    A basic neural network with configurable hidden layers and dropout
    for classification tasks in IoT anomaly detection.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize SimpleMLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def get_feature_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representation from the last hidden layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature representation before final classification layer
        """
        # Forward through all layers except the last one
        for layer in self.network[:-1]:
            x = layer(x)
        return x
    
    def get_parameters_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "SimpleMLP",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "total_parameters": self.get_parameters_count()
        }
    
    def freeze_layers(self, num_layers: int):
        """
        Freeze the first num_layers for transfer learning.
        
        Args:
            num_layers: Number of layers to freeze from the beginning
        """
        layer_count = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                if layer_count < num_layers:
                    for param in module.parameters():
                        param.requires_grad = False
                layer_count += 1
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
