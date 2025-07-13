"""
Unified Trust Strategy for TRUST_MCNet Flower Integration

This module combines trust-weighted aggregation with adaptive threshold adjustment
into a single, comprehensive strategy for federated learning.
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import deque
import numpy as np

import flwr as fl
from flwr.common import (
    Parameters, 
    FitRes, 
    EvaluateRes,
    parameters_to_ndarrays, 
    ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy

try:
    from ..trust_module.trust_evaluator import TrustEvaluator
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    
    # Add src directory to path if not already there
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator

logger = logging.getLogger(__name__)


class UnifiedTrustStrategy(fl.server.strategy.FedAvg):
    """
    Unified trust-weighted federated learning strategy with adaptive capabilities.
    
    This strategy combines:
    1. Trust-weighted aggregation using TRUST_MCNet evaluation mechanisms
    2. Adaptive threshold adjustment based on performance trends
    3. Comprehensive metrics tracking and monitoring
    
    The strategy can operate in two modes:
    - Standard Mode: Uses fixed trust thresholds (backward compatible)
    - Adaptive Mode: Dynamically adjusts thresholds based on performance
    
    This implementation follows SOLID principles:
    - Single Responsibility: Handles trust-aware aggregation with optional adaptation
    - Open/Closed: Extends FedAvg without modifying core logic
    - Liskov Substitution: Can replace any Flower strategy
    - Interface Segregation: Uses minimal interface from TrustEvaluator
    - Dependency Inversion: Depends on TrustEvaluator abstraction
    """

    def __init__(
        self,
        trust_evaluator: TrustEvaluator,
        # Adaptive parameters (None/False = disabled)
        enable_adaptation: bool = False,
        target_accuracy: float = 0.85,
        threshold_adaptation_rate: float = 0.05,
        max_threshold: float = 0.9,
        min_threshold: float = 0.3,
        performance_window: int = 5,
        adaptation_patience: int = 3,
        **fedavg_kwargs,
    ) -> None:
        """
        Initialize unified trust strategy.
        
        Args:
            trust_evaluator: TrustEvaluator instance for trust computation
            enable_adaptation: Whether to enable adaptive threshold adjustment
            target_accuracy: Target accuracy threshold for adaptation
            threshold_adaptation_rate: Rate of threshold adjustment
            max_threshold: Maximum allowed trust threshold
            min_threshold: Minimum allowed trust threshold
            performance_window: Number of rounds to consider for trends
            adaptation_patience: Rounds to wait before adapting threshold
            **fedavg_kwargs: All standard FedAvg parameters
        """
        # Initialize parent FedAvg with all standard parameters
        super().__init__(**fedavg_kwargs)
        
        # Store trust evaluator as long-lived object to maintain history
        self.trust_eval = trust_evaluator
        
        # Trust configuration
        self.trust_threshold = getattr(trust_evaluator, 'threshold', 0.5)
        
        # Adaptive configuration
        self.enable_adaptation = enable_adaptation
        self.target_accuracy = target_accuracy
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.performance_window = performance_window
        self.adaptation_patience = adaptation_patience
        
        # Adaptive state (only used if adaptation is enabled)
        if self.enable_adaptation:
            self.performance_history = deque(maxlen=performance_window)
            self.round_counter = 0
            self.last_adaptation_round = 0
            
            logger.info(f"Initialized UnifiedTrustStrategy with ADAPTIVE mode: "
                       f"target_accuracy={target_accuracy}, adaptation_rate={threshold_adaptation_rate}")
        else:
            logger.info(f"Initialized UnifiedTrustStrategy with STANDARD mode: "
                       f"threshold={self.trust_threshold}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """
        Aggregate client fit results using trust-weighted aggregation with optional adaptation.
        
        Args:
            server_round: Current round number
            results: List of (ClientProxy, FitRes) tuples from successful clients
            failures: List of failed client attempts
            
        Returns:
            Tuple of (aggregated_parameters, metrics_dict)
        """
        if failures and not self.accept_failures:
            return None, {}
            
        if not results:
            logger.warning("No client results to aggregate")
            return None, {}

        try:
            # Step 1: Perform trust-weighted aggregation
            aggregated_parameters, trust_metrics = self._perform_trust_aggregation(
                server_round, results
            )
            
            # Step 2: Handle adaptive threshold adjustment (if enabled)
            if self.enable_adaptation:
                self.round_counter = server_round
                
                # Collect round metrics for adaptation
                round_metrics = self._collect_round_metrics(results, trust_metrics)
                
                # Update performance history
                self.performance_history.append(round_metrics)
                
                # Adapt trust threshold if conditions are met
                if self._should_adapt_threshold():
                    old_threshold = self.trust_threshold
                    self._update_trust_threshold(round_metrics)
                    
                    if abs(self.trust_threshold - old_threshold) > 1e-6:
                        logger.info(f"Round {server_round}: Adapted trust threshold from "
                                   f"{old_threshold:.3f} to {self.trust_threshold:.3f}")
                        self.last_adaptation_round = server_round
                        
                        # Update trust evaluator threshold
                        self.trust_eval.threshold = self.trust_threshold
                
                # Add adaptation metrics
                trust_metrics.update({
                    'adaptive_trust_threshold': self.trust_threshold,
                    'target_accuracy': self.target_accuracy,
                    'performance_trend': self._calculate_performance_trend(),
                    'rounds_since_adaptation': server_round - self.last_adaptation_round,
                    'adaptation_enabled': True
                })
            else:
                trust_metrics['adaptation_enabled'] = False
            
            # Step 3: Add general metrics
            trust_metrics.update({
                'round': server_round,
                'strategy_type': 'unified_trust',
                'trust_mode': getattr(self.trust_eval, 'trust_mode', 'unknown')
            })
            
            return aggregated_parameters, trust_metrics
            
        except Exception as e:
            logger.error(f"Error in unified trust aggregation: {e}")
            # Graceful fallback to standard FedAvg
            logger.warning("Falling back to standard FedAvg aggregation")
            return super().aggregate_fit(server_round, results, failures)

    def _perform_trust_aggregation(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Parameters, Dict[str, Any]]:
        """
        Perform trust-weighted aggregation of client updates.
        
        Args:
            server_round: Current round number
            results: Client fit results
            
        Returns:
            Tuple of aggregated parameters and trust metrics
        """
        # Extract client information and model parameters
        client_ids = [r[0].cid for r in results]
        client_parameters = [parameters_to_ndarrays(r[1].parameters) for r in results]
        client_metrics = [r[1].metrics for r in results]
        
        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")
        
        # Convert parameters to format expected by TrustEvaluator
        client_updates_dict = {}
        param_names = [f"param_{j}" for j in range(len(client_parameters[0]))]
        
        for i, (client_id, params) in enumerate(zip(client_ids, client_parameters)):
            param_dict = {}
            for j, param_array in enumerate(params):
                param_name = param_names[j]
                param_dict[param_name] = self._numpy_to_torch(param_array)
            
            client_updates_dict[client_id] = param_dict
        
        # Evaluate trust scores for each client
        trust_scores = {}
        
        for i, (client_id, metrics) in enumerate(zip(client_ids, client_metrics)):
            try:
                # Extract performance metrics
                performance_metrics = {
                    'accuracy': metrics.get('accuracy', 0.5),
                    'loss': metrics.get('train_loss', 1.0),
                    'f1_score': metrics.get('f1_score', 0.5)
                }
                
                # Use TrustEvaluator's evaluate_trust method
                model_update = client_updates_dict[client_id]
                global_model = {}  # Placeholder
                
                trust_score = self.trust_eval.evaluate_trust(
                    client_id=client_id,
                    model_update=model_update,
                    performance_metrics=performance_metrics,
                    global_model=global_model,
                    round_number=server_round
                )
                
                trust_scores[client_id] = trust_score
                
            except Exception as e:
                logger.warning(f"Trust evaluation failed for client {client_id}: {e}")
                trust_scores[client_id] = 0.5
        
        # Delegate aggregation to TrustEvaluator with enhanced quarantine logic
        try:
            aggregated_params_torch, trust_statistics = self.trust_eval.aggregate_model_updates(
                client_updates=client_updates_dict,
                client_trust_scores=trust_scores,
                round_number=server_round,
                trim_ratio=0.1
            )
            
            # Convert back to NDArrays for Flower
            aggregated_ndarrays = []
            for j, param_name in enumerate(param_names):
                if param_name in aggregated_params_torch:
                    tensor_param = aggregated_params_torch[param_name]
                    numpy_param = tensor_param.detach().cpu().numpy()
                    aggregated_ndarrays.append(numpy_param)
                else:
                    # Fallback to averaging
                    param_arrays = [client_parameters[i][j] for i in range(len(client_parameters))]
                    avg_param = np.mean(param_arrays, axis=0)
                    aggregated_ndarrays.append(avg_param)
            
            # Convert to Flower Parameters
            aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
            
            # Extract trust vector and quarantine information
            trust_vector = [trust_scores[client_id] for client_id in client_ids]
            quarantined_clients = trust_statistics.get('quarantined_clients', [])
            surviving_clients = trust_statistics.get('surviving_clients', [])
            
        except Exception as e:
            logger.error(f"Trust evaluation failed: {e}")
            raise
        
        # Compute enhanced trust metrics including quarantine information
        trust_metrics = self._compute_trust_metrics(client_ids, trust_scores, trust_vector)
        
        # Add quarantine-specific metrics
        trust_metrics.update({
            'num_clients_used': len(trust_statistics.get('trusted_survivors', [])),
            'aggregation_method': 'trust_weighted_trimmed_mean_with_quarantine',
            'trust_threshold': self.trust_threshold,
            'quarantine_enabled': True,
            'quarantined_clients': quarantined_clients,
            'surviving_clients': surviving_clients,
            'num_quarantined': trust_statistics.get('num_quarantined', 0),
            'num_survivors': trust_statistics.get('num_survivors', 0),
            'quarantine_rate': trust_statistics.get('quarantine_rate', 0.0),
            'quarantine_stats': trust_statistics.get('quarantine_stats', {}),
            'trim_ratio': trust_statistics.get('trim_ratio', 0.1)
        })
        
        logger.info(f"Round {server_round}: Trust aggregation completed. "
                   f"Mean trust: {trust_metrics['mean_trust']:.3f}")
        
        return aggregated_parameters, trust_metrics

    def _collect_round_metrics(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        trust_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect round-level metrics for adaptation (only used if adaptation enabled)."""
        if not self.enable_adaptation:
            return {}
        
        if not results:
            return {
                'avg_accuracy': 0.0,
                'avg_loss': float('inf'),
                'num_clients': 0,
                'trust_scores': []
            }
        
        # Extract client metrics
        accuracies = []
        losses = []
        trust_scores = []
        
        for client_proxy, fit_res in results:
            if fit_res.metrics:
                if 'accuracy' in fit_res.metrics:
                    accuracies.append(fit_res.metrics['accuracy'])
                
                if 'train_loss' in fit_res.metrics:
                    losses.append(fit_res.metrics['train_loss'])
                elif 'loss' in fit_res.metrics:
                    losses.append(fit_res.metrics['loss'])
                
                if 'trust_score' in fit_res.metrics:
                    trust_scores.append(fit_res.metrics['trust_score'])
        
        # Compute aggregated metrics
        round_metrics = {
            'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'avg_loss': np.mean(losses) if losses else float('inf'),
            'num_clients': len(results),
            'trust_scores': trust_scores,
            'accuracy_std': np.std(accuracies) if len(accuracies) > 1 else 0.0,
            'loss_std': np.std(losses) if len(losses) > 1 else 0.0
        }
        
        # Add trust metrics
        round_metrics.update(trust_metrics)
        
        return round_metrics

    def _should_adapt_threshold(self) -> bool:
        """Determine if trust threshold should be adapted."""
        if not self.enable_adaptation:
            return False
        
        # Need sufficient history
        if len(self.performance_history) < 2:
            return False
        
        # Respect adaptation patience
        if self.round_counter - self.last_adaptation_round < self.adaptation_patience:
            return False
        
        return True

    def _update_trust_threshold(self, round_metrics: Dict[str, Any]) -> None:
        """Dynamically adjust trust threshold based on performance trends."""
        if not self.enable_adaptation:
            return
        
        current_accuracy = round_metrics.get('avg_accuracy', 0.0)
        performance_trend = self._calculate_performance_trend()
        
        # Calculate threshold adjustment
        threshold_delta = 0.0
        
        # Case 1: Below target accuracy - increase threshold to be more selective
        if current_accuracy < self.target_accuracy:
            threshold_delta = self.threshold_adaptation_rate
            logger.debug(f"Below target accuracy ({current_accuracy:.3f} < {self.target_accuracy:.3f}), "
                        f"increasing threshold")
        
        # Case 2: Above target with declining trend - increase threshold
        elif current_accuracy >= self.target_accuracy and performance_trend < -0.01:
            threshold_delta = self.threshold_adaptation_rate * 0.5
            logger.debug(f"Declining performance trend ({performance_trend:.3f}), "
                        f"moderately increasing threshold")
        
        # Case 3: Above target with stable/improving trend - decrease threshold
        elif current_accuracy > self.target_accuracy and performance_trend >= 0:
            threshold_delta = -self.threshold_adaptation_rate * 0.3
            logger.debug(f"Good performance with positive trend ({performance_trend:.3f}), "
                        f"slightly decreasing threshold")
        
        # Case 4: Significantly above target - decrease threshold more aggressively
        if current_accuracy > self.target_accuracy + 0.05:
            threshold_delta = -self.threshold_adaptation_rate * 0.5
            logger.debug(f"Significantly above target ({current_accuracy:.3f}), "
                        f"decreasing threshold to include more clients")
        
        # Apply threshold adjustment with bounds
        new_threshold = self.trust_threshold + threshold_delta
        self.trust_threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)
        
        logger.debug(f"Threshold adjustment: {threshold_delta:+.3f}, "
                    f"new threshold: {self.trust_threshold:.3f}")

    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend over the recent history window."""
        if not self.enable_adaptation or len(self.performance_history) < 2:
            return 0.0
        
        # Extract accuracies from history
        accuracies = [metrics.get('avg_accuracy', 0.0) for metrics in self.performance_history]
        
        if len(accuracies) < 2:
            return 0.0
        
        # Calculate linear trend using least squares
        x = np.arange(len(accuracies))
        y = np.array(accuracies)
        
        try:
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating performance trend: {e}")
            return 0.0

    def _numpy_to_torch(self, array: np.ndarray):
        """Convert numpy array to torch tensor with proper device handling."""
        try:
            import torch
            return torch.from_numpy(array).float()
        except Exception as e:
            logger.error(f"Failed to convert numpy to torch: {e}")
            return array

    def _compute_trust_metrics(
        self, 
        client_ids: List[str], 
        trust_scores: Dict[str, float],
        trust_vector: List[float]
    ) -> Dict[str, Any]:
        """Compute trust metrics for monitoring and logging."""
        if not trust_vector:
            return {
                'mean_trust': 0.0,
                'min_trust': 0.0,
                'max_trust': 0.0,
                'trust_std': 0.0,
                'trusted_clients_count': 0,
                'total_clients': len(client_ids)
            }
        
        return {
            'mean_trust': float(np.mean(trust_vector)),
            'min_trust': float(np.min(trust_vector)),
            'max_trust': float(np.max(trust_vector)),
            'trust_std': float(np.std(trust_vector)),
            'trusted_clients_count': sum(1 for score in trust_vector if score >= self.trust_threshold),
            'total_clients': len(client_ids)
        }

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and metrics."""
        if not self.enable_adaptation:
            return {
                'adaptation_enabled': False,
                'current_trust_threshold': self.trust_threshold
            }
        
        performance_trend = self._calculate_performance_trend()
        current_accuracy = (self.performance_history[-1].get('avg_accuracy', 0.0) 
                          if self.performance_history else 0.0)
        
        return {
            'adaptation_enabled': True,
            'current_trust_threshold': self.trust_threshold,
            'target_accuracy': self.target_accuracy,
            'current_accuracy': current_accuracy,
            'performance_trend': performance_trend,
            'rounds_since_adaptation': self.round_counter - self.last_adaptation_round,
            'performance_history_length': len(self.performance_history),
            'threshold_bounds': {
                'min': self.min_threshold,
                'max': self.max_threshold
            },
            'adaptation_config': {
                'adaptation_rate': self.threshold_adaptation_rate,
                'patience': self.adaptation_patience,
                'window_size': self.performance_window
            }
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        mode = "ADAPTIVE" if self.enable_adaptation else "STANDARD"
        return (f"UnifiedTrustStrategy({mode}: "
                f"trust_threshold={self.trust_threshold}, "
                f"trust_mode={getattr(self.trust_eval, 'trust_mode', 'unknown')}, "
                f"min_fit_clients={self.min_fit_clients})")


# Backward compatibility aliases
TrustWeightedStrategy = UnifiedTrustStrategy
AdaptiveTrustStrategy = lambda *args, **kwargs: UnifiedTrustStrategy(*args, enable_adaptation=True, **kwargs)
