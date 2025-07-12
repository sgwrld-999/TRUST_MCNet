"""
Model aggregation utilities for TRUST-MCNet federated learning.

This module implements robust aggregation methods including:
- Trust-weighted averaging
- Trimmed mean aggregation
- Gradient clipping and normalization
- Byzantine-robust aggregation techniques
- FLTrust aggregation for trusted root dataset
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Protocol, Any
import logging
from collections import OrderedDict
from abc import ABC, abstractmethod


class AggregatorInterface(ABC):
    """
    Base interface for federated learning aggregators.
    
    All aggregator implementations should inherit from this interface
    to ensure consistent API across different aggregation strategies.
    """
    
    @abstractmethod
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            weights: Optional weights for each client
            **kwargs: Additional aggregation-specific parameters
            
        Returns:
            Aggregated model parameters
        """
        pass


class FLTrustAggregator(AggregatorInterface):
    """
    FLTrust aggregation implementation based on "FLTrust: Byzantine-robust 
    Federated Learning via Trust Bootstrapping" (https://arxiv.org/pdf/2012.13995).
    
    Key features:
    - Uses a trusted root dataset to compute reference gradients
    - Computes cosine similarity trust scores between client and root updates
    - Clips trust scores to non-negative values
    - Normalizes client update magnitudes to match root update magnitude
    - Performs trust-weighted aggregation
    """
    
    def __init__(self, 
                 root_model: torch.nn.Module,
                 root_dataset_loader: Optional[torch.utils.data.DataLoader] = None,
                 server_learning_rate: float = 1.0,
                 clip_threshold: float = 10.0,
                 warm_up_rounds: int = 5,
                 trust_threshold: float = 0.0):
        """
        Initialize FLTrust aggregator.
        
        Args:
            root_model: Model for computing root updates (should be same architecture as clients)
            root_dataset_loader: DataLoader for trusted root dataset (can be None if manually provided)
            server_learning_rate: Global learning rate (η) for server updates
            clip_threshold: Maximum norm for gradient clipping
            warm_up_rounds: Number of rounds before trust scoring becomes effective
            trust_threshold: Minimum trust score threshold (scores below this are clipped to 0)
        """
        self.root_model = root_model
        self.root_dataset_loader = root_dataset_loader
        self.server_learning_rate = server_learning_rate
        self.clip_threshold = clip_threshold
        self.warm_up_rounds = warm_up_rounds
        self.trust_threshold = trust_threshold
        
        self.current_round = 0
        self.logger = logging.getLogger(__name__)
        
        # Store previous global model for computing updates
        self.previous_global_weights = None
        
        # Device management
        self.device = next(root_model.parameters()).device
        
    def set_root_dataset(self, root_dataset_loader: torch.utils.data.DataLoader):
        """Set or update the root dataset loader."""
        self.root_dataset_loader = root_dataset_loader
        
    def compute_root_update(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the trusted root update Δw_root from the root dataset.
        
        Args:
            global_weights: Current global model weights
            
        Returns:
            Root model update (Δw_root)
        """
        if self.root_dataset_loader is None:
            raise ValueError("Root dataset loader not provided. Use set_root_dataset() or provide in constructor.")
        
        # Set root model to current global weights
        self.root_model.load_state_dict(global_weights)
        self.root_model.train()
        
        # Compute gradient on root dataset
        root_gradients = {}
        total_loss = 0.0
        num_samples = 0
        
        # Zero gradients
        self.root_model.zero_grad()
        
        # Accumulate gradients over root dataset
        for batch_idx, (data, target) in enumerate(self.root_dataset_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.root_model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            
            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
        
        # Extract gradients as root update
        for name, param in self.root_model.named_parameters():
            if param.grad is not None:
                root_gradients[name] = param.grad.clone()
            else:
                root_gradients[name] = torch.zeros_like(param)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        self.logger.info(f"Root update computed from {num_samples} samples, avg loss: {avg_loss:.4f}")
        
        return root_gradients
    
    def compute_client_updates(self, client_weights: Dict[str, Dict[str, torch.Tensor]], 
                             global_weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute client updates Δw_k = w_k - w_t from client weights.
        
        Args:
            client_weights: Dictionary of client model weights
            global_weights: Current global model weights
            
        Returns:
            Dictionary of client updates
        """
        client_updates = {}
        
        for client_id, weights in client_weights.items():
            client_update = {}
            for param_name in weights.keys():
                if param_name in global_weights:
                    client_update[param_name] = weights[param_name] - global_weights[param_name]
                else:
                    self.logger.warning(f"Parameter {param_name} not found in global weights")
                    client_update[param_name] = weights[param_name]
            client_updates[client_id] = client_update
            
        return client_updates
    
    def compute_cosine_trust_scores(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                                  root_update: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute cosine similarity trust scores between client updates and root update.
        
        Args:
            client_updates: Dictionary of client model updates
            root_update: Root model update
            
        Returns:
            Dictionary of trust scores for each client
        """
        trust_scores = {}
        
        # Flatten root update to vector
        root_vector = torch.cat([param.flatten() for param in root_update.values()])
        root_norm = torch.norm(root_vector)
        
        if root_norm == 0:
            self.logger.warning("Root update has zero norm, using uniform trust scores")
            return {client_id: 1.0 for client_id in client_updates.keys()}
        
        for client_id, client_update in client_updates.items():
            # Flatten client update to vector
            client_vector = torch.cat([param.flatten() for param in client_update.values()])
            client_norm = torch.norm(client_vector)
            
            if client_norm == 0:
                trust_score = 0.0
            else:
                # Compute cosine similarity
                cosine_sim = torch.dot(client_vector, root_vector) / (client_norm * root_norm)
                trust_score = max(0.0, cosine_sim.item())  # Clip to non-negative
            
            # Apply trust threshold
            if trust_score < self.trust_threshold:
                trust_score = 0.0
                
            trust_scores[client_id] = trust_score
        
        return trust_scores
    
    def normalize_client_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                               root_update: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Normalize client update magnitudes to match root update magnitude.
        Δw_k ← (‖Δw_root‖₂ / ‖Δw_k‖₂) · Δw_k
        
        Args:
            client_updates: Dictionary of client model updates
            root_update: Root model update
            
        Returns:
            Dictionary of normalized client updates
        """
        normalized_updates = {}
        
        # Compute root update norm
        root_vector = torch.cat([param.flatten() for param in root_update.values()])
        root_norm = torch.norm(root_vector)
        
        if root_norm == 0:
            self.logger.warning("Root update has zero norm, skipping normalization")
            return client_updates
        
        for client_id, client_update in client_updates.items():
            # Compute client update norm
            client_vector = torch.cat([param.flatten() for param in client_update.values()])
            client_norm = torch.norm(client_vector)
            
            normalized_update = {}
            if client_norm == 0:
                # Keep zero update as is
                normalized_update = client_update
            else:
                # Normalize magnitude: scale factor = root_norm / client_norm
                scale_factor = root_norm / client_norm
                for param_name, param_value in client_update.items():
                    normalized_update[param_name] = scale_factor * param_value
            
            normalized_updates[client_id] = normalized_update
        
        return normalized_updates
    
    def apply_gradient_clipping(self, updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Apply gradient clipping to updates based on maximum norm.
        
        Args:
            updates: Dictionary of model updates
            
        Returns:
            Dictionary of clipped updates
        """
        clipped_updates = {}
        
        for client_id, update in updates.items():
            clipped_update = {}
            for param_name, param_value in update.items():
                param_norm = torch.norm(param_value)
                if param_norm > self.clip_threshold:
                    clipped_param = param_value * (self.clip_threshold / param_norm)
                    clipped_update[param_name] = clipped_param
                else:
                    clipped_update[param_name] = param_value
            clipped_updates[client_id] = clipped_update
            
        return clipped_updates
    
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None,
                 global_weights: Optional[Dict[str, torch.Tensor]] = None,
                 trust_scores_override: Optional[Dict[str, float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FLTrust algorithm.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters (full weights, not deltas)
            weights: Optional additional weights (e.g., data sizes) - will be combined with trust scores
            global_weights: Current global model weights (required for computing deltas)
            trust_scores_override: Optional custom trust scores to replace cosine similarity scores
            **kwargs: Additional parameters
            
        Returns:
            Aggregated global model update (to be added to current global weights)
        """
        if global_weights is None:
            raise ValueError("global_weights must be provided for FLTrust aggregation")
        
        self.current_round += 1
        
        # Step 1: Compute client updates (Δw_k = w_k - w_t)
        client_deltas = self.compute_client_updates(client_updates, global_weights)
        
        # Step 2: Compute root update from trusted dataset
        root_delta = self.compute_root_update(global_weights)
        
        # Step 3: Apply gradient clipping
        client_deltas = self.apply_gradient_clipping(client_deltas)
        
        # Step 4: Normalize client update magnitudes to match root update
        client_deltas = self.normalize_client_updates(client_deltas, root_delta)
        
        # Step 5: Compute trust scores
        if trust_scores_override is not None:
            trust_scores = trust_scores_override
            self.logger.info("Using provided trust scores override")
        else:
            trust_scores = self.compute_cosine_trust_scores(client_deltas, root_delta)
        
        # Step 6: Apply warm-up period (use uniform weights during initial rounds)
        if self.current_round <= self.warm_up_rounds:
            self.logger.info(f"Warm-up round {self.current_round}/{self.warm_up_rounds}, using uniform aggregation")
            trust_scores = {client_id: 1.0 for client_id in trust_scores.keys()}
        
        # Step 7: Combine with additional weights if provided
        if weights is not None:
            combined_weights = {}
            for client_id in trust_scores.keys():
                additional_weight = weights.get(client_id, 1.0)
                combined_weights[client_id] = trust_scores[client_id] * additional_weight
            trust_scores = combined_weights
        
        # Step 8: Normalize trust scores to sum to 1
        total_trust = sum(trust_scores.values())
        if total_trust > 0:
            normalized_trust_scores = {client_id: score / total_trust 
                                     for client_id, score in trust_scores.items()}
        else:
            # Fallback to uniform weights if all trust scores are zero
            num_clients = len(trust_scores)
            normalized_trust_scores = {client_id: 1.0 / num_clients 
                                     for client_id in trust_scores.keys()}
            self.logger.warning("All trust scores are zero, using uniform weights")
        
        # Step 9: Aggregate client updates: Δw_global = Σ_k (s_k / Σ_j s_j) · Δw_k
        param_names = list(next(iter(client_deltas.values())).keys())
        aggregated_delta = {}
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_id, client_delta in client_deltas.items():
                if param_name in client_delta:
                    client_weight = normalized_trust_scores[client_id]
                    weighted_param = client_weight * client_delta[param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            aggregated_delta[param_name] = weighted_sum if weighted_sum is not None else torch.zeros_like(
                next(iter(client_deltas.values()))[param_name])
        
        # Log aggregation statistics
        trust_summary = {k: f"{v:.3f}" for k, v in trust_scores.items()}
        self.logger.info(f"FLTrust Round {self.current_round}: Trust scores: {trust_summary}")
        self.logger.info(f"FLTrust aggregated {len(client_deltas)} client updates")
        
        return aggregated_delta
    
    def get_aggregation_info(self) -> Dict[str, Any]:
        """Get information about the aggregation state."""
        return {
            'aggregator_type': 'FLTrust',
            'current_round': self.current_round,
            'server_learning_rate': self.server_learning_rate,
            'clip_threshold': self.clip_threshold,
            'warm_up_rounds': self.warm_up_rounds,
            'trust_threshold': self.trust_threshold,
            'has_root_dataset': self.root_dataset_loader is not None
        }


class TrimmedMeanAggregator(AggregatorInterface):
    """
    Robust model aggregation using coordinate-wise trimmed mean to handle malicious clients.
    
    This aggregator applies trimmed mean to each coordinate independently, removing
    extreme values (outliers) before averaging. This provides robustness against
    Byzantine attacks and malicious model updates.
    
    The algorithm:
    1. For each parameter coordinate, collect values from all clients
    2. Sort the values for that coordinate  
    3. Discard the b smallest and b largest values
    4. Average the remaining values
    
    This is more robust than global trimming as it handles coordinate-wise attacks.
    """
    
    def __init__(self, trim_ratio: float = 0.1, clip_threshold: float = 5.0):
        """
        Initialize trimmed mean aggregator.
        
        Args:
            trim_ratio: Fraction of extreme values to trim (0.0 to 0.5)
                       For m clients, this removes trim_ratio*m smallest and largest values
            clip_threshold: Gradient clipping threshold for individual updates
        """
        self.trim_ratio = trim_ratio
        self.clip_threshold = clip_threshold
        self.logger = logging.getLogger(__name__)
        
        if not 0.0 <= trim_ratio <= 0.5:
            raise ValueError("trim_ratio must be between 0.0 and 0.5")
    
    def aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                 weights: Optional[Dict[str, float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using coordinate-wise trimmed mean.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            weights: Optional trust weights for each client (currently not used in trimmed mean)
            **kwargs: Additional parameters
            
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
        
        # Calculate trimming parameter
        m = len(clipped_updates)  # number of clients
        b = int(m * self.trim_ratio)  # number of values to trim from each side
        
        # Validate trimming parameter
        if b > m // 2:
            self.logger.warning(f"trim_ratio {self.trim_ratio} too large for {m} clients, "
                              f"reducing to maximum allowed")
            b = m // 2
        
        # Aggregate each parameter using coordinate-wise trimmed mean
        for param_name in param_names:
            # Collect parameter tensors from all clients
            param_tensors = []
            for client_id, update in clipped_updates.items():
                if param_name in update:
                    param_tensors.append(update[param_name])
            
            if param_tensors:
                # Apply coordinate-wise trimmed mean
                aggregated_param = self._coordinate_wise_trimmed_mean(param_tensors, b)
                aggregated_params[param_name] = aggregated_param
        
        self.logger.info(f"Aggregated {len(client_updates)} client updates using "
                        f"coordinate-wise trimmed mean (b={b})")
        return aggregated_params
    
    def _coordinate_wise_trimmed_mean(self, param_tensors: List[torch.Tensor], 
                                    b: int) -> torch.Tensor:
        """
        Apply coordinate-wise trimmed mean to a list of parameter tensors.
        
        Args:
            param_tensors: List of parameter tensors from different clients
            b: Number of extreme values to trim from each side for each coordinate
            
        Returns:
            Aggregated parameter tensor using coordinate-wise trimmed mean
        """
        if not param_tensors:
            raise ValueError("No parameter tensors provided")
        
        if len(param_tensors) == 1:
            return param_tensors[0]
        
        # Stack tensors along a new dimension (clients dimension)
        stacked = torch.stack(param_tensors, dim=0)  # Shape: (num_clients, *param_shape)
        original_shape = stacked.shape[1:]  # Original parameter shape
        
        # Flatten all dimensions except the client dimension
        flattened = stacked.view(stacked.shape[0], -1)  # Shape: (num_clients, num_coordinates)
        num_clients, num_coordinates = flattened.shape
        
        # Apply trimmed mean to each coordinate
        if b == 0:
            # No trimming - simple average
            result = torch.mean(flattened, dim=0)
        else:
            # Apply coordinate-wise trimming
            result = torch.zeros(num_coordinates, device=flattened.device, dtype=flattened.dtype)
            
            for coord_idx in range(num_coordinates):
                # Get values for this coordinate across all clients
                coord_values = flattened[:, coord_idx]  # Shape: (num_clients,)
                
                # Sort values for this coordinate
                sorted_values, _ = torch.sort(coord_values)
                
                # Trim b smallest and b largest values
                if 2 * b < num_clients:
                    trimmed_values = sorted_values[b:-b] if b > 0 else sorted_values
                    result[coord_idx] = torch.mean(trimmed_values)
                else:
                    # Fallback: if too few clients, take median
                    result[coord_idx] = torch.median(sorted_values)
        
        # Reshape back to original parameter shape
        return result.view(original_shape)
    
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


class FedAvgAggregator(AggregatorInterface):
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


class KrumAggregator(AggregatorInterface):
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
