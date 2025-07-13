"""
Test suite for unified trust strategy functionality.

This module tests the UnifiedTrustStrategy class which combines
trust-weighted aggregation with adaptive threshold adjustment.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple

# Import the unified strategy
try:
    from src.trust_mcnet.strategies.unified_trust_strategy import UnifiedTrustStrategy
except ImportError:
    pytest.skip("UnifiedTrustStrategy not available", allow_module_level=True)

# Import trust evaluator
try:
    from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
except ImportError:
    pytest.skip("TrustEvaluator not available", allow_module_level=True)


class TestUnifiedTrustStrategy:
    """Test cases for UnifiedTrustStrategy."""
    
    @pytest.fixture
    def mock_trust_evaluator(self):
        """Create a mock trust evaluator."""
        evaluator = Mock(spec=TrustEvaluator)
        evaluator.trust_mode = "hybrid"
        evaluator.threshold = 0.5
        evaluator.evaluate_trust.return_value = 0.7
        evaluator.aggregate_model_updates.return_value = ({}, [0.7, 0.8, 0.6])
        return evaluator
    
    @pytest.fixture
    def strategy_standard(self, mock_trust_evaluator):
        """Create a standard (non-adaptive) unified strategy."""
        return UnifiedTrustStrategy(
            trust_evaluator=mock_trust_evaluator,
            enable_adaptation=False,
            min_fit_clients=2,
            min_available_clients=2
        )
    
    @pytest.fixture
    def strategy_adaptive(self, mock_trust_evaluator):
        """Create an adaptive unified strategy."""
        return UnifiedTrustStrategy(
            trust_evaluator=mock_trust_evaluator,
            enable_adaptation=True,
            target_accuracy=0.85,
            threshold_adaptation_rate=0.05,
            min_fit_clients=2,
            min_available_clients=2
        )
    
    def test_initialization_standard_mode(self, mock_trust_evaluator):
        """Test strategy initialization in standard mode."""
        strategy = UnifiedTrustStrategy(
            trust_evaluator=mock_trust_evaluator,
            enable_adaptation=False
        )
        
        assert strategy.trust_eval == mock_trust_evaluator
        assert strategy.enable_adaptation is False
        assert strategy.trust_threshold == 0.5
        assert not hasattr(strategy, 'performance_history')  # Only exists if adaptive
    
    def test_initialization_adaptive_mode(self, mock_trust_evaluator):
        """Test strategy initialization in adaptive mode."""
        strategy = UnifiedTrustStrategy(
            trust_evaluator=mock_trust_evaluator,
            enable_adaptation=True,
            target_accuracy=0.9,
            threshold_adaptation_rate=0.1
        )
        
        assert strategy.trust_eval == mock_trust_evaluator
        assert strategy.enable_adaptation is True
        assert strategy.target_accuracy == 0.9
        assert strategy.threshold_adaptation_rate == 0.1
        assert hasattr(strategy, 'performance_history')
        assert len(strategy.performance_history) == 0
    
    @patch('flwr.common.parameters_to_ndarrays')
    @patch('flwr.common.ndarrays_to_parameters')
    def test_aggregate_fit_standard_mode(
        self, 
        mock_ndarrays_to_params,
        mock_params_to_ndarrays,
        strategy_standard
    ):
        """Test aggregation in standard mode."""
        # Mock Flower types
        from unittest.mock import MagicMock
        
        # Create mock client results
        mock_client_proxy = MagicMock()
        mock_client_proxy.cid = "client_1"
        
        mock_fit_res = MagicMock()
        mock_fit_res.parameters = MagicMock()
        mock_fit_res.metrics = {'accuracy': 0.8, 'train_loss': 0.3}
        
        results = [(mock_client_proxy, mock_fit_res)]
        
        # Mock parameter conversion
        mock_params_to_ndarrays.return_value = [np.array([1.0, 2.0])]
        mock_ndarrays_to_params.return_value = MagicMock()
        
        # Mock torch tensor creation
        with patch('torch.from_numpy') as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array([1.5, 2.5])
            mock_torch.return_value = mock_tensor
            
            # Configure trust evaluator mock
            strategy_standard.trust_eval.aggregate_model_updates.return_value = (
                {'param_0': mock_tensor}, [0.7]
            )
            
            # Test aggregation
            params, metrics = strategy_standard.aggregate_fit(1, results, [])
            
            # Verify results
            assert params is not None
            assert 'mean_trust' in metrics
            assert 'adaptation_enabled' in metrics
            assert metrics['adaptation_enabled'] is False
            assert metrics['strategy_type'] == 'unified_trust'
    
    def test_adaptation_status_standard_mode(self, strategy_standard):
        """Test adaptation status in standard mode."""
        status = strategy_standard.get_adaptation_status()
        
        assert status['adaptation_enabled'] is False
        assert 'current_trust_threshold' in status
        assert status['current_trust_threshold'] == 0.5
    
    def test_adaptation_status_adaptive_mode(self, strategy_adaptive):
        """Test adaptation status in adaptive mode."""
        status = strategy_adaptive.get_adaptation_status()
        
        assert status['adaptation_enabled'] is True
        assert status['target_accuracy'] == 0.85
        assert status['threshold_bounds']['min'] == 0.3
        assert status['threshold_bounds']['max'] == 0.9
        assert 'adaptation_config' in status
    
    def test_performance_trend_calculation(self, strategy_adaptive):
        """Test performance trend calculation."""
        # Add some performance history
        strategy_adaptive.performance_history.extend([
            {'avg_accuracy': 0.7},
            {'avg_accuracy': 0.75},
            {'avg_accuracy': 0.8}
        ])
        
        trend = strategy_adaptive._calculate_performance_trend()
        assert trend > 0  # Should be positive (improving)
        
        # Test declining trend
        strategy_adaptive.performance_history.clear()
        strategy_adaptive.performance_history.extend([
            {'avg_accuracy': 0.8},
            {'avg_accuracy': 0.75},
            {'avg_accuracy': 0.7}
        ])
        
        trend = strategy_adaptive._calculate_performance_trend()
        assert trend < 0  # Should be negative (declining)
    
    def test_threshold_adaptation(self, strategy_adaptive):
        """Test threshold adaptation logic."""
        initial_threshold = strategy_adaptive.trust_threshold
        
        # Test below target accuracy
        round_metrics = {'avg_accuracy': 0.7}  # Below target of 0.85
        strategy_adaptive.performance_history.extend([
            {'avg_accuracy': 0.6},
            {'avg_accuracy': 0.65}
        ])
        
        strategy_adaptive._update_trust_threshold(round_metrics)
        assert strategy_adaptive.trust_threshold > initial_threshold
        
        # Test above target accuracy with good trend
        strategy_adaptive.trust_threshold = 0.5  # Reset
        round_metrics = {'avg_accuracy': 0.9}  # Above target
        strategy_adaptive.performance_history.clear()
        strategy_adaptive.performance_history.extend([
            {'avg_accuracy': 0.85},
            {'avg_accuracy': 0.87},
            {'avg_accuracy': 0.9}
        ])
        
        strategy_adaptive._update_trust_threshold(round_metrics)
        assert strategy_adaptive.trust_threshold <= 0.5  # Should decrease or stay same
    
    def test_should_adapt_threshold(self, strategy_adaptive):
        """Test threshold adaptation conditions."""
        # Not enough history
        assert not strategy_adaptive._should_adapt_threshold()
        
        # Add sufficient history
        strategy_adaptive.performance_history.extend([
            {'avg_accuracy': 0.7},
            {'avg_accuracy': 0.75}
        ])
        strategy_adaptive.round_counter = 5
        strategy_adaptive.last_adaptation_round = 0
        
        assert strategy_adaptive._should_adapt_threshold()
        
        # Test patience (too soon after last adaptation)
        strategy_adaptive.last_adaptation_round = 3
        assert not strategy_adaptive._should_adapt_threshold()
    
    def test_backward_compatibility_aliases(self):
        """Test backward compatibility aliases."""
        from src.trust_mcnet.strategies.unified_trust_strategy import (
            TrustWeightedStrategy, 
            AdaptiveTrustStrategy
        )
        
        # These should be aliases to UnifiedTrustStrategy
        assert TrustWeightedStrategy == UnifiedTrustStrategy
        
        # AdaptiveTrustStrategy should be a lambda that enables adaptation
        # This is a more complex test since it's a lambda
        mock_evaluator = Mock()
        mock_evaluator.threshold = 0.5
        
        adaptive_strategy = AdaptiveTrustStrategy(trust_evaluator=mock_evaluator)
        assert adaptive_strategy.enable_adaptation is True
    
    def test_repr_string(self, strategy_standard, strategy_adaptive):
        """Test string representation."""
        repr_std = repr(strategy_standard)
        assert "UnifiedTrustStrategy(STANDARD" in repr_std
        
        repr_adp = repr(strategy_adaptive)
        assert "UnifiedTrustStrategy(ADAPTIVE" in repr_adp


if __name__ == "__main__":
    pytest.main([__file__])
