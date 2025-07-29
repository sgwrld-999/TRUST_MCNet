"""
Enhanced model implementations for TRUST-MCNet.

This module provides improved model architectures for TRUST-MCNet,
including advanced MLP and LSTM variants with attention mechanisms,
residual connections, and other enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any


class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow and training stability.
    
    Features:
    - Skip connections for better gradient flow
    - Batch normalization for training stability
    - Dropout for regularization
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout_rate: float = 0.2, 
        use_batch_norm: bool = True
    ):
        """
        Initialize residual block.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(ResidualBlock, self).__init__()
        
        # Linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
            
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection adapter if dimensions don't match
        self.use_adapter = (in_features != out_features)
        if self.use_adapter:
            self.adapter = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Save input for residual connection
        identity = x
        
        # Main path
        out = self.linear(x)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            # Handle both 2D and 3D inputs (for LSTM integration)
            if len(out.shape) == 3:
                # Reshape for BatchNorm1d
                batch_size, seq_len, features = out.shape
                out = out.reshape(-1, features)
                out = self.batch_norm(out)
                out = out.reshape(batch_size, seq_len, features)
            else:
                out = self.batch_norm(out)
                
        # Apply activation
        out = F.relu(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply residual connection
        if self.use_adapter:
            identity = self.adapter(identity)
            
        out = out + identity
        
        return out


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing dependencies in sequence data.
    
    Features:
    - Multi-head attention for capturing different relationship types
    - Scaled dot-product attention
    - Trainable projections for queries, keys, and values
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        """
        Initialize self-attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through self-attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to queries, keys, and values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Transpose and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(context)
        
        return output, attn_weights


class ImprovedMLP(nn.Module):
    """
    Improved MLP architecture with residual connections, batch normalization,
    and adaptive layer sizes.
    
    Features:
    - Configurable hidden layers
    - Residual connections for better gradient flow
    - Input normalization
    - Layer size adaptation for reducing overfitting
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize improved MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(ImprovedMLP, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Input normalization
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
            
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if use_residual:
                # Residual block
                layers.append(
                    ResidualBlock(
                        prev_dim, 
                        hidden_dim, 
                        dropout_rate=dropout_rate, 
                        use_batch_norm=use_batch_norm
                    )
                )
            else:
                # Traditional layer
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Combine hidden layers
        self.hidden_layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through improved MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Apply input normalization if enabled
        if self.use_batch_norm:
            if len(x.shape) == 3:
                # Handle 3D input (batch_size, seq_len, features)
                batch_size, seq_len, features = x.shape
                x = x.reshape(-1, features)
                x = self.input_norm(x)
                x = x.reshape(batch_size, seq_len, features)
            else:
                # Handle 2D input (batch_size, features)
                x = self.input_norm(x)
        
        # Forward through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        return self.output_layer(x)


class ImprovedLSTM(nn.Module):
    """
    Enhanced LSTM architecture with attention mechanism and residual connections.
    
    Features:
    - Bidirectional LSTM for capturing context from both directions
    - Self-attention for capturing dependencies between different time steps
    - Residual connections between LSTM layers
    - Configurable layer normalization and dropout
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        output_size: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize improved LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            output_size: Output dimension
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            use_attention: Whether to use self-attention mechanism
        """
        super(ImprovedLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        
        # Direction factor (1 for unidirectional, 2 for bidirectional)
        dir_factor = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size * dir_factor)
                for _ in range(num_layers)
            ])
            
        # Self-attention
        if use_attention:
            self.attention = SelfAttention(
                embed_dim=hidden_size * dir_factor,
                num_heads=4,
                dropout=dropout
            )
            
        # Output layer
        self.fc = nn.Linear(hidden_size * dir_factor, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through improved LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        # LSTM forward
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm[0](output)
            
        # Apply attention if enabled
        if self.use_attention:
            output, attention_weights = self.attention(output)
        
        # Take the output of the last time step
        # For bidirectional, concatenate forward and backward last outputs
        last_output = output[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Apply output layer
        result = self.fc(last_output)
        
        return result


class TemporalAttentionLSTM(nn.Module):
    """
    LSTM with temporal attention mechanism for time series data.
    
    Features:
    - Separate attention mechanism for temporal patterns
    - Weighted aggregation of time steps
    - Configurable attention heads
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        """
        Initialize temporal attention LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout probability
        """
        super(TemporalAttentionLSTM, self).__init__()
        
        self.bidirectional = bidirectional
        dir_factor = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention weights
        self.attention = nn.Linear(hidden_size * dir_factor, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * dir_factor, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal attention LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Calculate attention weights
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Apply attention weights
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_out                    # (batch_size, seq_len, hidden_size)
        ).squeeze(1)                    # (batch_size, hidden_size)
        
        # Apply dropout
        context = self.dropout(context)
        
        # Output layer
        output = self.fc(context)
        
        return output


def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function for creating models.
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "improved_mlp":
        return ImprovedMLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            dropout_rate=config.get('dropout_rate', 0.2),
            use_batch_norm=config.get('use_batch_norm', True),
            use_residual=config.get('use_residual', True)
        )
    elif model_type == "improved_lstm":
        return ImprovedLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config.get('num_layers', 2),
            output_size=config.get('output_size', 1),
            bidirectional=config.get('bidirectional', True),
            dropout=config.get('dropout', 0.2),
            use_layer_norm=config.get('use_layer_norm', True),
            use_attention=config.get('use_attention', True)
        )
    elif model_type == "temporal_attention_lstm":
        return TemporalAttentionLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_layers=config.get('num_layers', 2),
            bidirectional=config.get('bidirectional', True),
            dropout=config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
