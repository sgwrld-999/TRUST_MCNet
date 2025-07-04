"""
Refactored trust evaluation module using interface-based architecture.

This module implements trust evaluation mechanisms that conform to the
TrustEvaluatorInterface, providing:
- Registry pattern for trust evaluator selection
- Interface-based design for extensibility
- Production-grade error handling
- Comprehensive trust metrics
- Scalable trust computation
"""

import logging
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque
from abc import ABC
import numpy as np

from core.interfaces import TrustEvaluatorInterface
from core.abstractions import BaseTrustEvaluator
from core.exceptions import TrustEvaluationError, ConfigurationError
from core.types import ConfigType, ModelWeights, TrustScore, TrustMetrics

logger = logging.getLogger(__name__)


class CosineSimilarityTrustEvaluator(BaseTrustEvaluator):
    """
    Trust evaluator based on cosine similarity between model updates.
    
    Evaluates trust by computing cosine similarity between client updates
    and the global model or aggregated updates from other clients.
    """
    
    def __init__(self, config: ConfigType, **kwargs) -> None:
        """
        Initialize cosine similarity trust evaluator.
        
        Args:
            config: Trust evaluator configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        
        self.similarity_threshold = config.get('similarity_threshold', 0.5)
        self.use_global_reference = config.get('use_global_reference', True)
        self.window_size = config.get('history_window_size', 10)
        
        # History tracking
        self.global_update_history = deque(maxlen=self.window_size)
        self.client_similarity_history = defaultdict(lambda: deque(maxlen=self.window_size))
        
        logger.info(f"Initialized cosine similarity trust evaluator")
    
    def evaluate_trust(
        self,
        client_id: Optional[str] = None,
        client_update: Optional[ModelWeights] = None,
        round_num: int = 0,
        **kwargs
    ) -> TrustScore:
        """
        Evaluate trust based on cosine similarity.
        
        Args:
            client_id: Client identifier
            client_update: Client's model update
            round_num: Current round number
            **kwargs: Additional parameters
            
        Returns:
            Trust score between 0 and 1
            
        Raises:
            TrustEvaluationError: If evaluation fails
        """
        try:
            if client_update is None:
                raise TrustEvaluationError("Client update required for cosine similarity evaluation")
            
            # Convert update to vector for similarity computation
            update_vector = self._weights_to_vector(client_update)
            
            if len(self.global_update_history) == 0:
                # First round - assign neutral trust
                similarity = 0.5
                logger.debug(f"First round for client {client_id}, assigning neutral trust")
            else:
                # Compute similarity with reference
                if self.use_global_reference and self.global_update_history:
                    reference_vector = self.global_update_history[-1]
                    similarity = self._compute_cosine_similarity(update_vector, reference_vector)
                else:
                    # Use average similarity with recent updates
                    similarities = []
                    for ref_vector in self.global_update_history:
                        sim = self._compute_cosine_similarity(update_vector, ref_vector)
                        similarities.append(sim)
                    similarity = np.mean(similarities) if similarities else 0.5
            
            # Store similarity in history
            if client_id:
                self.client_similarity_history[client_id].append(similarity)
            
            # Convert similarity to trust score (normalize to [0, 1])
            trust_score = max(0.0, min(1.0, (similarity + 1) / 2))
            
            logger.debug(f"Client {client_id} trust score: {trust_score:.3f} (similarity: {similarity:.3f})")
            return trust_score
            
        except Exception as e:
            logger.error(f"Error evaluating cosine similarity trust: {e}")
            raise TrustEvaluationError(f"Failed to evaluate trust: {e}") from e
    
    def update_global_state(
        self,
        global_update: ModelWeights,
        round_num: int,
        **kwargs
    ) -> None:
        """
        Update global trust state with new global model update.
        
        Args:
            global_update: Global model update
            round_num: Current round number
            **kwargs: Additional parameters
        """
        try:
            global_vector = self._weights_to_vector(global_update)
            self.global_update_history.append(global_vector)
            logger.debug(f"Updated global state for round {round_num}")
            
        except Exception as e:
            logger.error(f"Error updating global trust state: {e}")
            raise TrustEvaluationError(f"Failed to update global state: {e}") from e
    
    def get_trust_metrics(self, client_id: Optional[str] = None) -> TrustMetrics:
        """
        Get comprehensive trust metrics.
        
        Args:
            client_id: Optional client ID for client-specific metrics
            
        Returns:
            Dictionary of trust metrics
        """
        metrics = {
            'evaluator_type': 'cosine_similarity',
            'similarity_threshold': self.similarity_threshold,
            'global_history_size': len(self.global_update_history),
            'total_clients_tracked': len(self.client_similarity_history)
        }
        
        if client_id and client_id in self.client_similarity_history:
            client_similarities = list(self.client_similarity_history[client_id])
            if client_similarities:
                metrics.update({
                    f'client_{client_id}_avg_similarity': np.mean(client_similarities),
                    f'client_{client_id}_std_similarity': np.std(client_similarities),
                    f'client_{client_id}_min_similarity': np.min(client_similarities),
                    f'client_{client_id}_max_similarity': np.max(client_similarities),
                    f'client_{client_id}_history_size': len(client_similarities)
                })
        
        return metrics
    
    def _weights_to_vector(self, weights: ModelWeights) -> np.ndarray:
        """
        Convert model weights to a single vector for similarity computation.
        
        Args:
            weights: Model weights (dict or list)
            
        Returns:
            Flattened weight vector
        """
        if isinstance(weights, dict):
            # Flatten all weight arrays and concatenate
            vectors = []
            for key in sorted(weights.keys()):  # Sort for consistency
                weight_array = weights[key]
                if hasattr(weight_array, 'flatten'):
                    vectors.append(weight_array.flatten())
                else:
                    vectors.append(np.array(weight_array).flatten())
            return np.concatenate(vectors)
        
        elif isinstance(weights, (list, tuple)):
            # Assume it's already a list of arrays
            vectors = []
            for weight_array in weights:
                if hasattr(weight_array, 'flatten'):
                    vectors.append(weight_array.flatten())
                else:
                    vectors.append(np.array(weight_array).flatten())
            return np.concatenate(vectors)
        
        else:
            # Assume it's already a vector
            return np.array(weights).flatten()
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity between -1 and 1
        """
        # Ensure vectors have the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class EntropyBasedTrustEvaluator(BaseTrustEvaluator):
    """
    Trust evaluator based on entropy of model predictions or updates.
    
    Evaluates trust by analyzing the entropy of client predictions
    or the distribution of model parameter updates.
    """
    
    def __init__(self, config: ConfigType, **kwargs) -> None:
        """
        Initialize entropy-based trust evaluator.
        
        Args:
            config: Trust evaluator configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        
        self.entropy_mode = config.get('entropy_mode', 'prediction')  # 'prediction' or 'weights'
        self.low_entropy_threshold = config.get('low_entropy_threshold', 0.1)
        self.high_entropy_threshold = config.get('high_entropy_threshold', 2.0)
        
        # History for entropy-based trust
        self.entropy_history = defaultdict(list)
        
        logger.info(f"Initialized entropy-based trust evaluator (mode: {self.entropy_mode})")
    
    def evaluate_trust(
        self,
        client_id: Optional[str] = None,
        client_update: Optional[ModelWeights] = None,
        predictions: Optional[np.ndarray] = None,
        round_num: int = 0,
        **kwargs
    ) -> TrustScore:
        """
        Evaluate trust based on entropy.
        
        Args:
            client_id: Client identifier
            client_update: Client's model update
            predictions: Client's predictions (for prediction mode)
            round_num: Current round number
            **kwargs: Additional parameters
            
        Returns:
            Trust score between 0 and 1
        """
        try:
            if self.entropy_mode == 'prediction':
                if predictions is None:
                    raise TrustEvaluationError("Predictions required for prediction entropy mode")
                entropy = self._compute_prediction_entropy(predictions)
            
            elif self.entropy_mode == 'weights':
                if client_update is None:
                    raise TrustEvaluationError("Client update required for weights entropy mode")
                entropy = self._compute_weight_entropy(client_update)
            
            else:
                raise TrustEvaluationError(f"Unknown entropy mode: {self.entropy_mode}")
            
            # Convert entropy to trust score
            trust_score = self._entropy_to_trust(entropy)
            
            # Store in history
            if client_id:
                self.entropy_history[client_id].append(entropy)
            
            logger.debug(f"Client {client_id} trust score: {trust_score:.3f} (entropy: {entropy:.3f})")
            return trust_score
            
        except Exception as e:
            logger.error(f"Error evaluating entropy-based trust: {e}")
            raise TrustEvaluationError(f"Failed to evaluate trust: {e}") from e
    
    def get_trust_metrics(self, client_id: Optional[str] = None) -> TrustMetrics:
        """
        Get entropy-based trust metrics.
        
        Args:
            client_id: Optional client ID for client-specific metrics
            
        Returns:
            Dictionary of trust metrics
        """
        metrics = {
            'evaluator_type': 'entropy_based',
            'entropy_mode': self.entropy_mode,
            'low_entropy_threshold': self.low_entropy_threshold,
            'high_entropy_threshold': self.high_entropy_threshold,
            'total_clients_tracked': len(self.entropy_history)
        }
        
        if client_id and client_id in self.entropy_history:
            client_entropies = self.entropy_history[client_id]
            if client_entropies:
                metrics.update({
                    f'client_{client_id}_avg_entropy': np.mean(client_entropies),
                    f'client_{client_id}_std_entropy': np.std(client_entropies),
                    f'client_{client_id}_min_entropy': np.min(client_entropies),
                    f'client_{client_id}_max_entropy': np.max(client_entropies)
                })
        
        return metrics
    
    def _compute_prediction_entropy(self, predictions: np.ndarray) -> float:
        """
        Compute entropy of prediction probabilities.
        
        Args:
            predictions: Prediction probabilities or logits
            
        Returns:
            Entropy value
        """
        # Convert to probabilities if needed
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            # Assume logits, convert to probabilities
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            predictions = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        # Compute entropy for each sample
        entropies = []
        for pred in predictions:
            # Add small epsilon to avoid log(0)
            pred = pred + 1e-10
            entropy = -np.sum(pred * np.log(pred))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def _compute_weight_entropy(self, weights: ModelWeights) -> float:
        """
        Compute entropy of weight distributions.
        
        Args:
            weights: Model weights
            
        Returns:
            Entropy value
        """
        all_weights = []
        
        if isinstance(weights, dict):
            for weight_array in weights.values():
                all_weights.extend(np.array(weight_array).flatten())
        else:
            for weight_array in weights:
                all_weights.extend(np.array(weight_array).flatten())
        
        all_weights = np.array(all_weights)
        
        # Create histogram of weights
        hist, _ = np.histogram(all_weights, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        
        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def _entropy_to_trust(self, entropy: float) -> float:
        """
        Convert entropy to trust score.
        
        Args:
            entropy: Entropy value
            
        Returns:
            Trust score between 0 and 1
        """
        # Low entropy (overly confident) and high entropy (random) both indicate low trust
        if entropy < self.low_entropy_threshold:
            # Too confident (possible overfitting or cheating)
            trust = entropy / self.low_entropy_threshold * 0.5
        elif entropy > self.high_entropy_threshold:
            # Too random (possible noise or poor training)
            excess = entropy - self.high_entropy_threshold
            trust = max(0.0, 0.5 - excess * 0.25)
        else:
            # Good entropy range - higher trust
            normalized = (entropy - self.low_entropy_threshold) / (self.high_entropy_threshold - self.low_entropy_threshold)
            trust = 0.5 + normalized * 0.5
        
        return max(0.0, min(1.0, trust))


class HybridTrustEvaluator(BaseTrustEvaluator):
    """
    Hybrid trust evaluator combining multiple trust mechanisms.
    
    Combines cosine similarity, entropy, and reputation-based trust
    evaluation for comprehensive client assessment.
    """
    
    def __init__(self, config: ConfigType, **kwargs) -> None:
        """
        Initialize hybrid trust evaluator.
        
        Args:
            config: Trust evaluator configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        
        # Initialize sub-evaluators
        cosine_config = config.get('cosine', {})
        entropy_config = config.get('entropy', {})
        
        self.cosine_evaluator = CosineSimilarityTrustEvaluator(cosine_config)
        self.entropy_evaluator = EntropyBasedTrustEvaluator(entropy_config)
        
        # Combination weights
        self.weights = config.get('weights', {
            'cosine': 0.4,
            'entropy': 0.3,
            'reputation': 0.3
        })
        
        # Reputation tracking
        self.reputation_history = defaultdict(lambda: deque(maxlen=config.get('reputation_window', 10)))
        self.reputation_decay = config.get('reputation_decay', 0.9)
        
        logger.info("Initialized hybrid trust evaluator")
    
    def evaluate_trust(
        self,
        client_id: Optional[str] = None,
        client_update: Optional[ModelWeights] = None,
        predictions: Optional[np.ndarray] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0,
        **kwargs
    ) -> TrustScore:
        """
        Evaluate trust using hybrid approach.
        
        Args:
            client_id: Client identifier
            client_update: Client's model update
            predictions: Client's predictions
            performance_metrics: Client's performance metrics
            round_num: Current round number
            **kwargs: Additional parameters
            
        Returns:
            Combined trust score
        """
        try:
            trust_components = {}
            
            # Cosine similarity trust
            if client_update is not None:
                trust_components['cosine'] = self.cosine_evaluator.evaluate_trust(
                    client_id=client_id,
                    client_update=client_update,
                    round_num=round_num
                )
            
            # Entropy trust
            if predictions is not None or client_update is not None:
                trust_components['entropy'] = self.entropy_evaluator.evaluate_trust(
                    client_id=client_id,
                    client_update=client_update,
                    predictions=predictions,
                    round_num=round_num
                )
            
            # Reputation trust
            if client_id:
                reputation_trust = self._compute_reputation_trust(client_id, performance_metrics)
                trust_components['reputation'] = reputation_trust
            
            # Combine trust scores
            combined_trust = self._combine_trust_scores(trust_components)
            
            logger.debug(f"Client {client_id} hybrid trust: {combined_trust:.3f} "
                        f"(components: {trust_components})")
            
            return combined_trust
            
        except Exception as e:
            logger.error(f"Error evaluating hybrid trust: {e}")
            raise TrustEvaluationError(f"Failed to evaluate trust: {e}") from e
    
    def update_global_state(
        self,
        global_update: ModelWeights,
        round_num: int,
        **kwargs
    ) -> None:
        """
        Update global state for all sub-evaluators.
        
        Args:
            global_update: Global model update
            round_num: Current round number
            **kwargs: Additional parameters
        """
        self.cosine_evaluator.update_global_state(global_update, round_num, **kwargs)
    
    def get_trust_metrics(self, client_id: Optional[str] = None) -> TrustMetrics:
        """
        Get comprehensive hybrid trust metrics.
        
        Args:
            client_id: Optional client ID for client-specific metrics
            
        Returns:
            Combined trust metrics
        """
        metrics = {
            'evaluator_type': 'hybrid',
            'combination_weights': self.weights
        }
        
        # Add sub-evaluator metrics
        cosine_metrics = self.cosine_evaluator.get_trust_metrics(client_id)
        entropy_metrics = self.entropy_evaluator.get_trust_metrics(client_id)
        
        for key, value in cosine_metrics.items():
            metrics[f'cosine_{key}'] = value
        
        for key, value in entropy_metrics.items():
            metrics[f'entropy_{key}'] = value
        
        # Add reputation metrics
        if client_id and client_id in self.reputation_history:
            rep_history = list(self.reputation_history[client_id])
            if rep_history:
                metrics.update({
                    f'client_{client_id}_avg_reputation': np.mean(rep_history),
                    f'client_{client_id}_std_reputation': np.std(rep_history),
                    f'client_{client_id}_reputation_trend': rep_history[-1] - rep_history[0] if len(rep_history) > 1 else 0
                })
        
        return metrics
    
    def _compute_reputation_trust(
        self,
        client_id: str,
        performance_metrics: Optional[Dict[str, float]]
    ) -> float:
        """
        Compute reputation-based trust score.
        
        Args:
            client_id: Client identifier
            performance_metrics: Client's performance metrics
            
        Returns:
            Reputation trust score
        """
        if performance_metrics is None:
            # Use historical average if no current metrics
            if client_id in self.reputation_history:
                history = list(self.reputation_history[client_id])
                return np.mean(history) if history else 0.5
            return 0.5
        
        # Compute current reputation based on performance
        accuracy = performance_metrics.get('accuracy', 0.0)
        loss = performance_metrics.get('loss', float('inf'))
        
        # Convert metrics to reputation score
        reputation = accuracy
        if loss != float('inf') and loss > 0:
            # Lower loss is better
            reputation = (reputation + (1.0 / (1.0 + loss))) / 2
        
        # Apply decay to historical reputation and add current
        if client_id in self.reputation_history:
            history = list(self.reputation_history[client_id])
            if history:
                decayed_avg = np.mean(history) * self.reputation_decay
                reputation = (decayed_avg + reputation) / 2
        
        # Store in history
        self.reputation_history[client_id].append(reputation)
        
        return reputation
    
    def _combine_trust_scores(self, trust_components: Dict[str, float]) -> float:
        """
        Combine individual trust scores using weighted average.
        
        Args:
            trust_components: Dictionary of trust scores
            
        Returns:
            Combined trust score
        """
        if not trust_components:
            return 0.5  # Neutral trust if no components
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for component, score in trust_components.items():
            weight = self.weights.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.mean(list(trust_components.values()))
        
        return weighted_sum / total_weight


# Trust Evaluator Registry
class TrustEvaluatorRegistry:
    """Registry for trust evaluators."""
    
    _evaluators = {
        'cosine': CosineSimilarityTrustEvaluator,
        'entropy': EntropyBasedTrustEvaluator,
        'hybrid': HybridTrustEvaluator,
    }
    
    @classmethod
    def register(cls, name: str, evaluator_class: type) -> None:
        """Register a new trust evaluator."""
        if not issubclass(evaluator_class, TrustEvaluatorInterface):
            raise ValueError(f"Evaluator {evaluator_class} must implement TrustEvaluatorInterface")
        cls._evaluators[name] = evaluator_class
        logger.info(f"Registered trust evaluator: {name}")
    
    @classmethod
    def create(cls, name: str, config: ConfigType, **kwargs) -> TrustEvaluatorInterface:
        """Create a trust evaluator instance."""
        if name not in cls._evaluators:
            available = ', '.join(cls._evaluators.keys())
            raise TrustEvaluationError(f"Trust evaluator '{name}' not found. Available: {available}")
        
        evaluator_class = cls._evaluators[name]
        return evaluator_class(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available trust evaluators."""
        return list(cls._evaluators.keys())


def create_trust_evaluator(name: str, config: ConfigType, **kwargs) -> TrustEvaluatorInterface:
    """
    Factory function to create trust evaluator instances.
    
    Args:
        name: Trust evaluator name
        config: Trust evaluator configuration
        **kwargs: Additional parameters
        
    Returns:
        Trust evaluator instance
    """
    return TrustEvaluatorRegistry.create(name, config, **kwargs)


# Export public interface
__all__ = [
    'CosineSimilarityTrustEvaluator',
    'EntropyBasedTrustEvaluator',
    'HybridTrustEvaluator',
    'TrustEvaluatorRegistry',
    'create_trust_evaluator'
]
