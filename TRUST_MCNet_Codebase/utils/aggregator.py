"""
Model aggregation utilities for TRUST-MCNet federated learning.

This module implements robust aggregation methods including:
- Trust-weighted averaging
- Trimmed mean aggregation
- Gradient clipping and normalization
- Byzantine-robust aggregation techniques
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from collections import OrderedDict


class TrimmedMeanAggregator:
    """
    Robust model aggregation using trimmed mean to handle malicious clients.
    
    This aggregator removes extreme values (outliers) before averaging,
    making it resistant to Byzantine attacks and malicious model updates.
    """
    
    def __init__(self, trim_ratio: float = 0.1, clip_threshold: float = 5.0):
        """
        Initialize trimmed mean aggregator.
        
        Args:
            trim_ratio: Fraction of extreme values to trim (0.0 to 0.5)
            clip_threshold: Gradient clipping threshold for individual updates
        """
        self.trim_ratio = trim_ratio
        self.clip_threshold = clip_threshold
        self.logger = logging.getLogger(__name__)
        
        if not 0.0 <= trim_ratio <= 0.5:
            raise ValueError("trim_ratio must be between 0.0 and 0.5")
    
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using trimmed mean with optional weighting.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            weights: Optional trust weights for each client
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Apply gradient clipping to individual updates
        clipped_updates = self._clip_gradients(client_updates)
        
        # Get parameter names from first client
        param_names = list(next(iter(clipped_updates.values())).keys())
        aggregated_params = OrderedDict()
        
        # Aggregate each parameter separately
        for param_name in param_names:
            param_values = []
            client_weights = []
            
            for client_id, update in clipped_updates.items():
                if param_name in update:
                    param_values.append(update[param_name])
                    client_weights.append(weights.get(client_id, 1.0) if weights else 1.0)
            
            if param_values:
                # Apply trimmed mean aggregation
                aggregated_param = self._trimmed_mean_aggregate(
                    param_values, client_weights
                )
                aggregated_params[param_name] = aggregated_param
        
        self.logger.info(f"Aggregated {len(client_updates)} client updates using trimmed mean")
        return aggregated_params
    
    def _clip_gradients(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Apply gradient clipping to client updates to prevent extreme values.
        
        Args:
            client_updates: Raw client model updates
            
        Returns:
            Gradient-clipped client updates
        """
        clipped_updates = {}
        
        for client_id, update in client_updates.items():
            clipped_update = {}
            
            for param_name, param_tensor in update.items():
                # Calculate L2 norm of the parameter
                param_norm = torch.norm(param_tensor)
                
                # Apply clipping if necessary
                if param_norm > self.clip_threshold:
                    clipped_param = param_tensor * (self.clip_threshold / param_norm)
                    clipped_update[param_name] = clipped_param
                else:
                    clipped_update[param_name] = param_tensor
            
            clipped_updates[client_id] = clipped_update
        
        return clipped_updates
    
    def _trimmed_mean_aggregate(self, param_values: List[torch.Tensor], 
                              weights: List[float]) -> torch.Tensor:
        """
        Compute trimmed mean of parameter values with optional weighting.
        
        Args:
            param_values: List of parameter tensors from different clients
            weights: List of weights corresponding to each parameter
            
        Returns:
            Aggregated parameter tensor
        """
        if len(param_values) == 1:
            return param_values[0]
        
        # Stack parameters along new dimension for processing
        stacked_params = torch.stack(param_values, dim=0)
        weights_tensor = torch.tensor(weights, dtype=stacked_params.dtype, 
                                     device=stacked_params.device)
        
        # Apply trimming
        num_clients = len(param_values)
        num_trim = int(num_clients * self.trim_ratio)
        
        if num_trim > 0:
            # Calculate weighted norms for each client's contribution
            norms = torch.norm(stacked_params.view(num_clients, -1), dim=1)
            weighted_norms = norms * weights_tensor
            
            # Find indices to trim (both extremes)
            num_trim_each_side = num_trim // 2
            _, sorted_indices = torch.sort(weighted_norms)
            
            # Remove extreme values
            keep_indices = sorted_indices[num_trim_each_side:num_clients - num_trim_each_side]
            
            trimmed_params = stacked_params[keep_indices]
            trimmed_weights = weights_tensor[keep_indices]
        else:
            trimmed_params = stacked_params
            trimmed_weights = weights_tensor
        
        # Compute weighted average
        if trimmed_weights.sum() > 0:
            normalized_weights = trimmed_weights / trimmed_weights.sum()
            aggregated = torch.sum(trimmed_params * normalized_weights.view(-1, *([1] * (trimmed_params.dim() - 1))), dim=0)
        else:
            # Fallback to simple average if all weights are zero
            aggregated = torch.mean(trimmed_params, dim=0)
        
        return aggregated


class FedAvgAggregator:
    """
    Standard FedAvg aggregation with trust-based weighting.
    
    Implements the classic federated averaging algorithm with optional
    trust-based client weighting for improved robustness.
    """
    
    def __init__(self, clip_threshold: float = 10.0):
        """
        Initialize FedAvg aggregator.
        
        Args:
            clip_threshold: Gradient clipping threshold
        """
        self.clip_threshold = clip_threshold
        self.logger = logging.getLogger(__name__)
    
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None,
                 data_sizes: Optional[Dict[str, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            weights: Optional trust weights for each client
            data_sizes: Optional data sizes for each client (for standard FedAvg weighting)
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Determine aggregation weights
        aggregation_weights = self._compute_aggregation_weights(
            client_updates, weights, data_sizes
        )
        
        # Get parameter names
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = OrderedDict()
        
        # Aggregate each parameter
        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0
            
            for client_id, update in client_updates.items():
                if param_name in update:
                    param_tensor = update[param_name]
                    client_weight = aggregation_weights[client_id]
                    
                    # Apply gradient clipping
                    param_norm = torch.norm(param_tensor)
                    if param_norm > self.clip_threshold:
                        param_tensor = param_tensor * (self.clip_threshold / param_norm)
                    
                    # Weighted accumulation
                    if weighted_sum is None:
                        weighted_sum = client_weight * param_tensor
                    else:
                        weighted_sum += client_weight * param_tensor
                    
                    total_weight += client_weight
            
            # Normalize by total weight
            if total_weight > 0:
                aggregated_params[param_name] = weighted_sum / total_weight
            else:
                # Fallback to first client's parameters
                aggregated_params[param_name] = next(iter(client_updates.values()))[param_name]
        
        self.logger.info(f"Aggregated {len(client_updates)} client updates using FedAvg")
        return aggregated_params
    
    def _compute_aggregation_weights(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                                   trust_weights: Optional[Dict[str, float]] = None,
                                   data_sizes: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Compute aggregation weights combining trust and data size information.
        
        Args:
            client_updates: Client model updates
            trust_weights: Trust scores for each client
            data_sizes: Data sizes for each client
            
        Returns:
            Final aggregation weights for each client
        """
        client_ids = list(client_updates.keys())
        aggregation_weights = {}
        
        for client_id in client_ids:
            # Start with uniform weight
            weight = 1.0
            
            # Apply trust weighting if available
            if trust_weights and client_id in trust_weights:
                weight *= trust_weights[client_id]
            
            # Apply data size weighting if available (standard FedAvg)
            if data_sizes and client_id in data_sizes:
                weight *= data_sizes[client_id]
            
            aggregation_weights[client_id] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(aggregation_weights.values())
        if total_weight > 0:
            for client_id in aggregation_weights:
                aggregation_weights[client_id] /= total_weight
        else:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(client_ids)
            aggregation_weights = {client_id: uniform_weight for client_id in client_ids}
        
        return aggregation_weights


class KrumAggregator:
    """
    Krum aggregation for Byzantine-robust federated learning.
    
    Selects the most representative client update based on distance
    to other updates, providing robustness against malicious clients.
    """
    
    def __init__(self, num_malicious: int = 1):
        """
        Initialize Krum aggregator.
        
        Args:
            num_malicious: Expected number of malicious clients
        """
        self.num_malicious = num_malicious
        self.logger = logging.getLogger(__name__)
    
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate using Krum algorithm.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            weights: Optional weights (not used in standard Krum)
            
        Returns:
            Selected client's model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        if len(client_updates) <= 2 * self.num_malicious:
            self.logger.warning("Not enough clients for robust Krum aggregation")
            # Fallback to simple averaging
            return self._simple_average(client_updates)
        
        # Flatten all client updates for distance calculation
        client_vectors = {}
        for client_id, update in client_updates.items():
            flattened = torch.cat([param.flatten() for param in update.values()])
            client_vectors[client_id] = flattened
        
        # Calculate Krum scores
        client_ids = list(client_vectors.keys())
        scores = {}
        
        for i, client_id in enumerate(client_ids):
            distances = []
            client_vector = client_vectors[client_id]
            
            for j, other_id in enumerate(client_ids):
                if i != j:
                    other_vector = client_vectors[other_id]
                    distance = torch.norm(client_vector - other_vector, p=2).item()
                    distances.append(distance)
            
            # Sort distances and sum the closest (n - f - 1) distances
            distances.sort()
            n_closest = len(client_ids) - self.num_malicious - 1
            score = sum(distances[:n_closest])
            scores[client_id] = score
        
        # Select client with minimum Krum score
        selected_client = min(scores.keys(), key=lambda x: scores[x])
        
        self.logger.info(f"Krum selected client {selected_client} "
                        f"with score {scores[selected_client]:.4f}")
        
        return client_updates[selected_client]
    
    def _simple_average(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Fallback simple averaging when Krum is not applicable."""
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = OrderedDict()
        
        for param_name in param_names:
            param_sum = None
            count = 0
            
            for update in client_updates.values():
                if param_name in update:
                    if param_sum is None:
                        param_sum = update[param_name].clone()
                    else:
                        param_sum += update[param_name]
                    count += 1
            
            if count > 0:
                aggregated_params[param_name] = param_sum / count
        
        return aggregated_params
