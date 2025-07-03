"""
Unit tests for model functionality.
"""

import unittest
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model import MLP, LSTM


class TestModelFunctionality(unittest.TestCase):
    """Test suite for model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.batch_size = 32
        self.input_dim = 784  # MNIST-like
        self.hidden_dims = [128, 64]
        self.output_dim = 10
        self.seq_len = 20
    
    def test_mlp_creation(self):
        """Test MLP model creation."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Check model exists
        self.assertIsInstance(model, nn.Module)
        
        # Check parameter count (rough estimation)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Create test input
        x = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_mlp_gradient_computation(self):
        """Test MLP gradient computation."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Create test input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_lstm_creation(self):
        """Test LSTM model creation."""
        model = LSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=self.output_dim
        )
        
        # Check model exists
        self.assertIsInstance(model, nn.Module)
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        hidden_dim = 128
        num_layers = 2
        
        model = LSTM(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=self.output_dim
        )
        
        # Create test input (batch_size, seq_len, input_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_lstm_gradient_computation(self):
        """Test LSTM gradient computation."""
        model = LSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=self.output_dim
        )
        
        # Create test input and target
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))
        
        # Forward pass
        output = model(x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_model_mode_switching(self):
        """Test switching between train and eval modes."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Test initial mode (should be train)
        self.assertTrue(model.training)
        
        # Switch to eval mode
        model.eval()
        self.assertFalse(model.training)
        
        # Switch back to train mode
        model.train()
        self.assertTrue(model.training)
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Test CPU (default)
        device = torch.device('cpu')
        model = model.to(device)
        x = torch.randn(self.batch_size, self.input_dim).to(device)
        
        output = model(x)
        self.assertEqual(output.device, device)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            x = x.to(device)
            
            output = model(x)
            self.assertEqual(output.device, device)
    
    def test_model_parameter_shapes(self):
        """Test that model parameters have expected shapes."""
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Get parameter shapes
        param_shapes = [p.shape for p in model.parameters()]
        
        # First layer: input_dim -> hidden_dims[0]
        expected_first_weight = (self.hidden_dims[0], self.input_dim)
        expected_first_bias = (self.hidden_dims[0],)
        
        # Check that we have expected parameter shapes
        self.assertIn(expected_first_weight, param_shapes)
        self.assertIn(expected_first_bias, param_shapes)
        
        # Last layer: hidden_dims[-1] -> output_dim
        expected_last_weight = (self.output_dim, self.hidden_dims[-1])
        expected_last_bias = (self.output_dim,)
        
        self.assertIn(expected_last_weight, param_shapes)
        self.assertIn(expected_last_bias, param_shapes)


class TestModelRobustness(unittest.TestCase):
    """Test suite for model robustness and edge cases."""
    
    def test_empty_hidden_dims(self):
        """Test MLP with empty hidden dimensions (direct input to output)."""
        model = MLP(
            input_dim=10,
            hidden_dims=[],
            output_dim=5
        )
        
        x = torch.randn(8, 10)
        output = model(x)
        
        # Should work as a linear layer
        self.assertEqual(output.shape, (8, 5))
    
    def test_single_hidden_layer(self):
        """Test MLP with single hidden layer."""
        model = MLP(
            input_dim=10,
            hidden_dims=[20],
            output_dim=5
        )
        
        x = torch.randn(8, 10)
        output = model(x)
        
        self.assertEqual(output.shape, (8, 5))
    
    def test_lstm_single_layer(self):
        """Test LSTM with single layer."""
        model = LSTM(
            input_dim=10,
            hidden_dim=20,
            num_layers=1,
            output_dim=5
        )
        
        x = torch.randn(8, 15, 10)  # (batch, seq, features)
        output = model(x)
        
        self.assertEqual(output.shape, (8, 5))
    
    def test_model_with_small_batch(self):
        """Test models with very small batch sizes."""
        mlp_model = MLP(input_dim=10, hidden_dims=[20], output_dim=5)
        lstm_model = LSTM(input_dim=10, hidden_dim=20, num_layers=1, output_dim=5)
        
        # Test with batch size 1
        x_mlp = torch.randn(1, 10)
        x_lstm = torch.randn(1, 5, 10)
        
        output_mlp = mlp_model(x_mlp)
        output_lstm = lstm_model(x_lstm)
        
        self.assertEqual(output_mlp.shape, (1, 5))
        self.assertEqual(output_lstm.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
