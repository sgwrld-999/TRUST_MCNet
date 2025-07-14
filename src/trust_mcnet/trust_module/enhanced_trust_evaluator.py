"""
Enhanced Trust Evaluation Module for TRUST-MCNet

This module provides a refactored trust evaluation system with improved:
- Code organization and readability
- Type safety and error handling
- Separation of concerns
- Documentation and maintainability
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy


class TrustMode(Enum):
    """Enumeration of supported trust evaluation modes."""
    COSINE = "cosine"
    ENTROPY = "entropy"
    REPUTATION = "reputation"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class TrustMetrics:
    """Immutable container for individual trust metric components."""
    cosine_similarity: float
    entropy_score: float
    reputation_score: float
    timestamp: float = field(default_factory=lambda: torch.tensor(0.0).item())
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'cosine_similarity': self.cosine_similarity,
            'entropy_score': self.entropy_score,
            'reputation_score': self.reputation_score,
            'timestamp': self.timestamp
        }


@dataclass(frozen=True)
class TrustScore:
    """Immutable container for final trust evaluation results."""
    overall_score: float
    component_scores: TrustMetrics
    client_id: str
    round_number: int
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate trust score after initialization."""
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError(f"Trust score must be between 0 and 1, got {self.overall_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class ClientUpdate:
    """Container for client update information."""
    model_parameters: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, float]
    client_model: Optional[torch.nn.Module] = None
    participation_rate: float = 1.0
    anomaly_flags: int = 0
    
    def get_flattened_parameters(self) -> torch.Tensor:
        """Get flattened parameter vector for similarity calculations."""
        flattened = []
        for param in self.model_parameters.values():
            flattened.append(param.detach().cpu().flatten())
        return torch.cat(flattened)


@dataclass
class EvaluationContext:
    """Container for evaluation context information."""
    global_model: Dict[str, torch.Tensor]
    round_number: int
    global_update_avg: Optional[Dict[str, torch.Tensor]] = None
    probe_data: Optional[torch.utils.data.DataLoader] = None
    
    def get_global_flattened_parameters(self) -> torch.Tensor:
        """Get flattened global model parameters."""
        flattened = []
        for param in self.global_model.values():
            flattened.append(param.detach().cpu().flatten())
        return torch.cat(flattened)


class TrustCalculationError(Exception):
    """Raised when trust calculation fails."""
    pass


class TrustCalculator(ABC):
    """Abstract base class for trust calculation strategies."""
    
    @abstractmethod
    def calculate_trust(
        self, 
        client_update: ClientUpdate, 
        context: EvaluationContext
    ) -> float:
        """Calculate trust score for a client update."""
        pass


class CosineSimilarityCalculator(TrustCalculator):
    """Calculates trust based on cosine similarity between client and global updates."""
    
    def __init__(self, similarity_threshold: float = 0.1):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_trust(
        self, 
        client_update: ClientUpdate, 
        context: EvaluationContext
    ) -> float:
        """
        Calculate trust based on cosine similarity.
        
        Implements: cos_i^t = cos(Δw_i^t, Δw̄^t)
        
        Args:
            client_update: Client's model update
            context: Evaluation context with global information
            
        Returns:
            Cosine similarity-based trust score (0-1)
        """
        try:
            client_params = client_update.get_flattened_parameters()
            
            if context.global_update_avg is not None:
                # Use average of all client updates
                global_avg_flattened = self._flatten_parameters(context.global_update_avg)
                reference_params = global_avg_flattened
            else:
                # Fallback to global model parameters
                reference_params = context.get_global_flattened_parameters()
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                client_params.unsqueeze(0), 
                reference_params.unsqueeze(0)
            ).item()
            
            # Convert to trust score (higher similarity = higher trust)
            trust_score = max(0.0, (similarity + 1.0) / 2.0)  # Map [-1,1] to [0,1]
            
            self.logger.debug(f"Cosine similarity: {similarity:.4f}, trust: {trust_score:.4f}")
            return trust_score
            
        except Exception as e:
            self.logger.warning(f"Cosine trust calculation failed: {e}")
            raise TrustCalculationError(f"Failed to calculate cosine trust: {e}") from e
    
    def _flatten_parameters(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten parameter dictionary to single tensor."""
        flattened = []
        for param in params.values():
            flattened.append(param.detach().cpu().flatten())
        return torch.cat(flattened)


class EntropyBasedCalculator(TrustCalculator):
    """Calculates trust based on prediction entropy on probe dataset."""
    
    def __init__(self, max_entropy_estimate: float = 2.3):  # ln(10) for 10 classes
        self.max_entropy_estimate = max_entropy_estimate
        self.logger = logging.getLogger(__name__)
    
    def calculate_trust(
        self, 
        client_update: ClientUpdate, 
        context: EvaluationContext
    ) -> float:
        """
        Calculate trust based on prediction entropy.
        
        Args:
            client_update: Client's model update
            context: Evaluation context with probe data
            
        Returns:
            Entropy-based trust score (0-1)
        """
        try:
            if context.probe_data is None or client_update.client_model is None:
                return self._fallback_parameter_entropy(client_update)
            
            return self._calculate_prediction_entropy(client_update.client_model, context.probe_data)
            
        except Exception as e:
            self.logger.warning(f"Entropy trust calculation failed: {e}")
            return 0.5  # Neutral trust on error
    
    def _calculate_prediction_entropy(
        self, 
        model: torch.nn.Module, 
        probe_data: torch.utils.data.DataLoader
    ) -> float:
        """Calculate entropy of model predictions on probe data."""
        model.eval()
        entropies = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(probe_data):
                if batch_idx >= 10:  # Limit computation
                    break
                
                try:
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Calculate entropy for each sample
                    epsilon = 1e-10
                    sample_entropies = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
                    entropies.extend(sample_entropies.cpu().numpy())
                    
                except Exception as e:
                    self.logger.debug(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if not entropies:
            return 0.5
        
        # Calculate expected entropy and normalize
        expected_entropy = np.mean(entropies)
        normalized_entropy = expected_entropy / self.max_entropy_estimate
        
        # Transform to trust score (moderate entropy = higher trust)
        optimal_entropy = 0.5
        entropy_deviation = abs(normalized_entropy - optimal_entropy)
        trust_score = max(0.0, 1.0 - 2 * entropy_deviation)
        
        return trust_score
    
    def _fallback_parameter_entropy(self, client_update: ClientUpdate) -> float:
        """Fallback entropy calculation using parameter distributions."""
        try:
            entropies = []
            
            for param_name, param_tensor in client_update.model_parameters.items():
                param_flat = param_tensor.detach().cpu().numpy().flatten()
                
                if len(param_flat) == 0 or np.std(param_flat) < 1e-8:
                    continue
                
                # Create histogram and calculate entropy
                n_bins = min(50, max(10, len(param_flat) // 20))
                hist, _ = np.histogram(param_flat, bins=n_bins, density=True)
                hist = hist / (hist.sum() + 1e-10) + 1e-10
                
                param_entropy = -np.sum(hist * np.log(hist))
                entropies.append(param_entropy)
            
            if not entropies:
                return 0.5
            
            avg_entropy = np.mean(entropies)
            return min(1.0, avg_entropy / 5.0)  # Normalize roughly
            
        except Exception:
            return 0.5


class ReputationCalculator(TrustCalculator):
    """Calculates trust based on historical performance reputation."""
    
    def __init__(self, window_size: int = 10, decay_factor: float = 0.9):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.client_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_trust(
        self, 
        client_update: ClientUpdate, 
        context: EvaluationContext
    ) -> float:
        """
        Calculate trust based on historical performance.
        
        Args:
            client_update: Client's model update
            context: Evaluation context
            
        Returns:
            Reputation-based trust score (0-1)
        """
        try:
            # Extract performance metrics
            accuracy = client_update.performance_metrics.get('accuracy', 0.5)
            loss = client_update.performance_metrics.get('loss', 1.0)
            participation = client_update.participation_rate
            flags = client_update.anomaly_flags
            
            # Calculate current performance score
            current_score = self._calculate_performance_score(accuracy, loss, participation, flags)
            
            # Update history (this modifies state - consider if this is appropriate)
            client_id = f"client_{id(client_update)}"  # Temporary ID
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            
            self.client_history[client_id].append(current_score)
            
            # Keep only recent history
            if len(self.client_history[client_id]) > self.window_size:
                self.client_history[client_id] = self.client_history[client_id][-self.window_size:]
            
            # Calculate weighted reputation
            history = self.client_history[client_id]
            if len(history) == 1:
                return current_score
            
            # Apply exponential decay to historical scores
            weights = [self.decay_factor ** i for i in range(len(history) - 1, -1, -1)]
            weighted_sum = sum(score * weight for score, weight in zip(history, weights))
            weight_sum = sum(weights)
            
            reputation_score = weighted_sum / weight_sum if weight_sum > 0 else current_score
            
            return max(0.0, min(1.0, reputation_score))
            
        except Exception as e:
            self.logger.warning(f"Reputation trust calculation failed: {e}")
            return 0.5
    
    def _calculate_performance_score(
        self, 
        accuracy: float, 
        loss: float, 
        participation: float, 
        flags: int
    ) -> float:
        """Calculate performance score from metrics."""
        # Normalize accuracy (assume it's already in [0,1])
        acc_score = max(0.0, min(1.0, accuracy))
        
        # Normalize loss (inverse relationship)
        loss_score = max(0.0, min(1.0, 1.0 / (1.0 + loss)))
        
        # Participation penalty
        participation_score = max(0.0, min(1.0, participation))
        
        # Anomaly penalty
        anomaly_penalty = max(0.0, min(1.0, 1.0 - flags * 0.1))
        
        # Weighted combination
        performance_score = (
            0.4 * acc_score + 
            0.3 * loss_score + 
            0.2 * participation_score + 
            0.1 * anomaly_penalty
        )
        
        return performance_score


class EnhancedTrustEvaluator:
    """
    Enhanced trust evaluation system with improved architecture.
    
    Features:
    - Clear separation of concerns
    - Configurable trust calculation strategies
    - Comprehensive error handling
    - Type-safe interfaces
    - Extensive logging and monitoring
    """
    
    def __init__(
        self,
        trust_mode: Union[str, TrustMode] = TrustMode.HYBRID,
        threshold: float = 0.5,
        learning_rate: float = 0.01,
        use_dynamic_weights: bool = True,
        probe_data: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Convert string to enum if necessary
        if isinstance(trust_mode, str):
            trust_mode = TrustMode(trust_mode)
        
        self.trust_mode = trust_mode
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.use_dynamic_weights = use_dynamic_weights
        self.probe_data = probe_data
        self.config = config or {}
        
        # Initialize calculators
        self.cosine_calculator = CosineSimilarityCalculator()
        self.entropy_calculator = EntropyBasedCalculator()
        self.reputation_calculator = ReputationCalculator()
        
        # Dynamic weight adaptation
        self.theta = np.array([0.4, 0.3, 0.3])  # [cosine, entropy, reputation]
        self.theta_history = [self.theta.copy()]
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized EnhancedTrustEvaluator with mode: {trust_mode.value}")
    
    def evaluate_trust(
        self,
        client_id: str,
        client_update: ClientUpdate,
        context: EvaluationContext
    ) -> TrustScore:
        """
        Main entry point for trust evaluation.
        
        Args:
            client_id: Unique identifier for the client
            client_update: Client's update information
            context: Evaluation context
            
        Returns:
            Comprehensive trust score with component details
        """
        try:
            if self.trust_mode == TrustMode.COSINE:
                overall_score = self.cosine_calculator.calculate_trust(client_update, context)
                component_scores = TrustMetrics(overall_score, 0.0, 0.0)
                
            elif self.trust_mode == TrustMode.ENTROPY:
                overall_score = self.entropy_calculator.calculate_trust(client_update, context)
                component_scores = TrustMetrics(0.0, overall_score, 0.0)
                
            elif self.trust_mode == TrustMode.REPUTATION:
                overall_score = self.reputation_calculator.calculate_trust(client_update, context)
                component_scores = TrustMetrics(0.0, 0.0, overall_score)
                
            else:  # HYBRID
                component_scores = self._calculate_all_metrics(client_update, context)
                overall_score = self._combine_trust_metrics(component_scores)
            
            return TrustScore(
                overall_score=overall_score,
                component_scores=component_scores,
                client_id=client_id,
                round_number=context.round_number
            )
            
        except Exception as e:
            self.logger.error(f"Trust evaluation failed for client {client_id}: {e}")
            # Return neutral trust on error
            neutral_metrics = TrustMetrics(0.5, 0.5, 0.5)
            return TrustScore(
                overall_score=0.5,
                component_scores=neutral_metrics,
                client_id=client_id,
                round_number=context.round_number,
                confidence=0.0  # Low confidence due to error
            )
    
    def _calculate_all_metrics(
        self, 
        client_update: ClientUpdate, 
        context: EvaluationContext
    ) -> TrustMetrics:
        """Calculate all trust metric components."""
        cosine_score = self.cosine_calculator.calculate_trust(client_update, context)
        entropy_score = self.entropy_calculator.calculate_trust(client_update, context)
        reputation_score = self.reputation_calculator.calculate_trust(client_update, context)
        
        return TrustMetrics(
            cosine_similarity=cosine_score,
            entropy_score=entropy_score,
            reputation_score=reputation_score
        )
    
    def _combine_trust_metrics(self, metrics: TrustMetrics) -> float:
        """Combine individual trust metrics into overall score."""
        if self.use_dynamic_weights:
            # Use adaptive weights
            weights = self.theta
        else:
            # Use static weights
            weights = np.array([0.4, 0.3, 0.3])
        
        overall_score = (
            weights[0] * metrics.cosine_similarity +
            weights[1] * metrics.entropy_score +
            weights[2] * metrics.reputation_score
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def update_dynamic_weights(
        self, 
        accuracy_delta: float, 
        round_number: int
    ) -> None:
        """Update dynamic weights based on performance feedback."""
        if not self.use_dynamic_weights:
            return
        
        # Simplified ρ-adaptive update rule
        # In practice, this would be more sophisticated
        gradient = np.array([accuracy_delta, accuracy_delta * 0.5, accuracy_delta * 0.3])
        self.theta += self.learning_rate * gradient
        
        # Ensure weights sum to 1 and are non-negative
        self.theta = np.maximum(0, self.theta)
        self.theta = self.theta / np.sum(self.theta)
        
        self.theta_history.append(self.theta.copy())
        
        self.logger.debug(f"Updated weights: {self.theta}")
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """Get summary of trust evaluator state."""
        return {
            'trust_mode': self.trust_mode.value,
            'threshold': self.threshold,
            'current_weights': self.theta.tolist(),
            'use_dynamic_weights': self.use_dynamic_weights,
            'weight_history_length': len(self.theta_history)
        }


# Factory function for backward compatibility
def create_trust_evaluator(**kwargs) -> EnhancedTrustEvaluator:
    """Factory function to create trust evaluator with backward compatibility."""
    return EnhancedTrustEvaluator(**kwargs)
