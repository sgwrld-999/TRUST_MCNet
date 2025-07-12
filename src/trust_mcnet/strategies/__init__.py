"""
Federated learning strategies using the new interface-based architecture.

This module implements various federated learning strategies that conform
to the StrategyInterface, providing:
- Registry pattern for strategy selection
- Interface-based design for extensibility
- Production-grade error handling
- Comprehensive configuration support
- Trust-aware aggregation
"""

import logging
from abc import ABC
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict

# Import framework-specific modules when available
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg, FedAdam, FedProx
    from flwr.common import (
        Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
        ndarrays_to_parameters, parameters_to_ndarrays
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

from core.interfaces import StrategyInterface, TrustEvaluatorInterface
from core.abstractions import BaseStrategy
from core.exceptions import StrategyError, TrustEvaluationError
from core.types import ConfigType, ModelWeights, ClientResults, AggregationResult

logger = logging.getLogger(__name__)


class FederatedAveragingStrategy(BaseStrategy):
    """
    Federated Averaging (FedAvg) strategy implementation.
    
    Implements the classic FedAvg algorithm with optional trust-based
    client selection and weighted aggregation.
    """
    
    def __init__(
        self, 
        config: ConfigType,
        trust_evaluator: Optional[TrustEvaluatorInterface] = None,
        **kwargs
    ) -> None:
        """
        Initialize FedAvg strategy.
        
        Args:
            config: Strategy configuration
            trust_evaluator: Optional trust evaluator for client selection
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, **kwargs)
        self.trust_evaluator = trust_evaluator
        self.min_fit_clients = config.get('min_fit_clients', 2)
        self.min_eval_clients = config.get('min_eval_clients', 2)
        self.min_available_clients = config.get('min_available_clients', 2)
        self.fraction_fit = config.get('fraction_fit', 1.0)
        self.fraction_eval = config.get('fraction_eval', 1.0)
        
        # Trust-based selection parameters
        self.use_trust = trust_evaluator is not None
        self.trust_threshold = config.get('trust_threshold', 0.5)
        
        logger.info(f"Initialized FedAvg strategy with trust: {self.use_trust}")
    
    def configure_fit(self, round_num: int, **kwargs) -> Dict[str, Any]:
        """
        Configure the fit round.
        
        Args:
            round_num: Current round number
            **kwargs: Additional configuration parameters
            
        Returns:
            Configuration for the fit round
        """
        return {
            "round": round_num,
            "epochs": self.config.get('local_epochs', 1),
            "batch_size": self.config.get('batch_size', 32),
            "learning_rate": self.config.get('learning_rate', 0.01)
        }
    
    def configure_evaluate(self, round_num: int, **kwargs) -> Dict[str, Any]:
        """
        Configure the evaluation round.
        
        Args:
            round_num: Current round number
            **kwargs: Additional configuration parameters
            
        Returns:
            Configuration for the evaluation round
        """
        return {
            "round": round_num,
            "batch_size": self.config.get('eval_batch_size', 64)
        }
    
    def aggregate_fit(
        self, 
        round_num: int,
        results: List[ClientResults],
        **kwargs
    ) -> AggregationResult:
        """
        Aggregate client fit results using weighted averaging.
        
        Args:
            round_num: Current round number
            results: List of client fit results
            **kwargs: Additional parameters
            
        Returns:
            Aggregated model weights and metrics
            
        Raises:
            StrategyError: If aggregation fails
        """
        try:
            if not results:
                raise StrategyError("No client results to aggregate")
            
            # Extract weights and sample counts
            weights_list = []
            sample_counts = []
            
            for result in results:
                if 'weights' not in result or 'num_samples' not in result:
                    raise StrategyError("Invalid client result format")
                
                weights_list.append(result['weights'])
                sample_counts.append(result['num_samples'])
            
            # Apply trust-based filtering if enabled
            if self.use_trust and self.trust_evaluator:
                weights_list, sample_counts = self._apply_trust_filtering(
                    weights_list, sample_counts, round_num
                )
            
            # Perform weighted aggregation
            aggregated_weights = self._weighted_average(weights_list, sample_counts)
            
            # Calculate aggregation metrics
            total_samples = sum(sample_counts)
            num_clients_used = len(weights_list)
            
            metrics = {
                'total_samples': total_samples,
                'num_clients_used': num_clients_used,
                'aggregation_method': 'weighted_average'
            }
            
            if self.use_trust:
                metrics['trust_filtering_enabled'] = True
                metrics['trust_threshold'] = self.trust_threshold
            
            return {
                'weights': aggregated_weights,
                'metrics': metrics,
                'num_clients': num_clients_used
            }
            
        except Exception as e:
            logger.error(f"Error in aggregate_fit: {e}")
            raise StrategyError(f"Failed to aggregate fit results: {e}") from e
    
    def aggregate_evaluate(
        self,
        round_num: int,
        results: List[ClientResults],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Aggregate client evaluation results.
        
        Args:
            round_num: Current round number
            results: List of client evaluation results
            **kwargs: Additional parameters
            
        Returns:
            Aggregated evaluation metrics
        """
        try:
            if not results:
                return {'loss': float('inf'), 'accuracy': 0.0, 'num_clients': 0}
            
            total_loss = 0.0
            total_accuracy = 0.0
            total_samples = 0
            
            for result in results:
                samples = result.get('num_samples', 1)
                total_loss += result.get('loss', 0.0) * samples
                total_accuracy += result.get('accuracy', 0.0) * samples
                total_samples += samples
            
            avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
            avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
            
            return {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'num_clients': len(results),
                'total_samples': total_samples
            }
            
        except Exception as e:
            logger.error(f"Error in aggregate_evaluate: {e}")
            raise StrategyError(f"Failed to aggregate evaluation results: {e}") from e
    
    def _apply_trust_filtering(
        self,
        weights_list: List[ModelWeights],
        sample_counts: List[int],
        round_num: int
    ) -> Tuple[List[ModelWeights], List[int]]:
        """
        Apply trust-based filtering to client updates.
        
        Args:
            weights_list: List of client model weights
            sample_counts: List of client sample counts
            round_num: Current round number
            
        Returns:
            Filtered weights and sample counts
        """
        try:
            if not self.trust_evaluator:
                return weights_list, sample_counts
            
            # Evaluate trust scores for all clients
            trust_scores = []
            for weights in weights_list:
                score = self.trust_evaluator.evaluate_trust(
                    client_update=weights,
                    round_num=round_num
                )
                trust_scores.append(score)
            
            # Filter clients based on trust threshold
            filtered_weights = []
            filtered_counts = []
            
            for i, (weights, count, score) in enumerate(zip(weights_list, sample_counts, trust_scores)):
                if score >= self.trust_threshold:
                    filtered_weights.append(weights)
                    filtered_counts.append(count)
                else:
                    logger.debug(f"Filtered out client {i} with trust score {score:.3f}")
            
            if not filtered_weights:
                logger.warning(f"All clients filtered out, using top client")
                # Use the client with highest trust score
                best_idx = np.argmax(trust_scores)
                filtered_weights = [weights_list[best_idx]]
                filtered_counts = [sample_counts[best_idx]]
            
            logger.info(f"Trust filtering: {len(filtered_weights)}/{len(weights_list)} clients selected")
            return filtered_weights, filtered_counts
            
        except Exception as e:
            logger.error(f"Error in trust filtering: {e}")
            # Fallback to no filtering
            return weights_list, sample_counts
    
    def _weighted_average(
        self,
        weights_list: List[ModelWeights],
        sample_counts: List[int]
    ) -> ModelWeights:
        """
        Compute weighted average of model weights.
        
        Args:
            weights_list: List of client model weights
            sample_counts: List of client sample counts
            
        Returns:
            Aggregated model weights
        """
        if not weights_list:
            raise StrategyError("Cannot compute weighted average of empty weights list")
        
        total_samples = sum(sample_counts)
        
        # Initialize aggregated weights with zeros
        aggregated = None
        
        for weights, count in zip(weights_list, sample_counts):
            weight_factor = count / total_samples
            
            if aggregated is None:
                # Initialize with first client's weights
                if isinstance(weights, dict):
                    aggregated = {k: v * weight_factor for k, v in weights.items()}
                else:
                    aggregated = [w * weight_factor for w in weights]
            else:
                # Add weighted contributions
                if isinstance(weights, dict):
                    for k, v in weights.items():
                        aggregated[k] += v * weight_factor
                else:
                    for i, w in enumerate(weights):
                        aggregated[i] += w * weight_factor
        
        return aggregated


class FederatedAdamStrategy(FederatedAveragingStrategy):
    """
    Federated Adam (FedAdam) strategy implementation.
    
    Extends FedAvg with adaptive moment estimation for server-side optimization.
    """
    
    def __init__(
        self,
        config: ConfigType,
        trust_evaluator: Optional[TrustEvaluatorInterface] = None,
        **kwargs
    ) -> None:
        """
        Initialize FedAdam strategy.
        
        Args:
            config: Strategy configuration
            trust_evaluator: Optional trust evaluator
            **kwargs: Additional parameters
        """
        super().__init__(config, trust_evaluator, **kwargs)
        
        # FedAdam specific parameters
        self.server_learning_rate = config.get('server_learning_rate', 1.0)
        self.server_momentum = config.get('server_momentum', 0.9)
        self.beta_1 = config.get('beta_1', 0.9)
        self.beta_2 = config.get('beta_2', 0.999)
        self.epsilon = config.get('epsilon', 1e-8)
        
        # Initialize server-side optimizer state
        self.m_t = None  # First moment
        self.v_t = None  # Second moment
        self.t = 0       # Time step
        
        logger.info("Initialized FedAdam strategy")
    
    def aggregate_fit(
        self,
        round_num: int,
        results: List[ClientResults],
        **kwargs
    ) -> AggregationResult:
        """
        Aggregate using FedAdam with server-side adaptive optimization.
        
        Args:
            round_num: Current round number
            results: Client fit results
            **kwargs: Additional parameters
            
        Returns:
            Aggregated result with FedAdam optimization
        """
        # Get standard weighted average
        fed_avg_result = super().aggregate_fit(round_num, results, **kwargs)
        
        # Apply FedAdam server optimization
        aggregated_weights = self._apply_fed_adam(fed_avg_result['weights'])
        
        fed_avg_result['weights'] = aggregated_weights
        fed_avg_result['metrics']['aggregation_method'] = 'fed_adam'
        fed_avg_result['metrics']['server_learning_rate'] = self.server_learning_rate
        
        return fed_avg_result
    
    def _apply_fed_adam(self, weights: ModelWeights) -> ModelWeights:
        """
        Apply FedAdam server-side optimization.
        
        Args:
            weights: Model weights from aggregation
            
        Returns:
            Optimized weights
        """
        self.t += 1
        
        if self.m_t is None:
            # Initialize moments
            if isinstance(weights, dict):
                self.m_t = {k: np.zeros_like(v) for k, v in weights.items()}
                self.v_t = {k: np.zeros_like(v) for k, v in weights.items()}
            else:
                self.m_t = [np.zeros_like(w) for w in weights]
                self.v_t = [np.zeros_like(w) for w in weights]
        
        # Update moments and apply optimization
        if isinstance(weights, dict):
            optimized_weights = {}
            for k, w in weights.items():
                # Update biased first moment estimate
                self.m_t[k] = self.beta_1 * self.m_t[k] + (1 - self.beta_1) * w
                
                # Update biased second raw moment estimate
                self.v_t[k] = self.beta_2 * self.v_t[k] + (1 - self.beta_2) * (w ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m_t[k] / (1 - self.beta_1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v_t[k] / (1 - self.beta_2 ** self.t)
                
                # Update weights
                optimized_weights[k] = w - self.server_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            optimized_weights = []
            for i, w in enumerate(weights):
                # Update moments
                self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * w
                self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * (w ** 2)
                
                # Bias correction
                m_hat = self.m_t[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v_t[i] / (1 - self.beta_2 ** self.t)
                
                # Update
                optimized_weights.append(w - self.server_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return optimized_weights


# Strategy Registry
class StrategyRegistry:
    """Registry for federated learning strategies."""
    
    _strategies = {
        'fedavg': FederatedAveragingStrategy,
        'fedadam': FederatedAdamStrategy,
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a new strategy."""
        if not issubclass(strategy_class, StrategyInterface):
            raise ValueError(f"Strategy {strategy_class} must implement StrategyInterface")
        cls._strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: ConfigType,
        trust_evaluator: Optional[TrustEvaluatorInterface] = None,
        **kwargs
    ) -> StrategyInterface:
        """Create a strategy instance."""
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise StrategyError(f"Strategy '{name}' not found. Available: {available}")
        
        strategy_class = cls._strategies[name]
        return strategy_class(config, trust_evaluator=trust_evaluator, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available strategies."""
        return list(cls._strategies.keys())


def create_strategy(
    name: str,
    config: ConfigType,
    trust_evaluator: Optional[TrustEvaluatorInterface] = None,
    **kwargs
) -> StrategyInterface:
    """
    Factory function to create strategy instances.
    
    Args:
        name: Strategy name
        config: Strategy configuration
        trust_evaluator: Optional trust evaluator
        **kwargs: Additional parameters
        
    Returns:
        Strategy instance
    """
    return StrategyRegistry.create(name, config, trust_evaluator, **kwargs)


# Export public interface
__all__ = [
    'FederatedAveragingStrategy',
    'FederatedAdamStrategy', 
    'StrategyRegistry',
    'create_strategy'
]
