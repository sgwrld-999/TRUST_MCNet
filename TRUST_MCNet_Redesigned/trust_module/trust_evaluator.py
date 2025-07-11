"""
Trust evaluation module for TRUST-MCNet federated learning framework.

This module implements various trust evaluation mechanisms including:
- Cosine similarity-based trust
- Entropy-based trust evaluation  
- Reputation-based trust scoring
- Hybrid trust combination methods
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Union, Tuple, Optional
from scipy.stats import entropy, spearmanr
from collections import defaultdict
import logging
import warnings


class TrustEvaluator:
    """
    Comprehensive trust evaluation system for federated learning clients.
    
    Supports multiple trust evaluation modes:
    - 'cosine': Cosine similarity between model updates
    - 'entropy': Entropy-based trust evaluation
    - 'reputation': Historical performance-based reputation
    - 'hybrid': Combination of multiple trust metrics
    """
    
    def __init__(self, trust_mode: str = 'hybrid', threshold: float = 0.5, 
                 learning_rate: float = 0.01, use_dynamic_weights: bool = True,
                 probe_data: Optional[torch.utils.data.DataLoader] = None):
        """
        Initialize trust evaluator.
        
        Args:
            trust_mode: Trust evaluation method ('cosine', 'entropy', 'reputation', 'hybrid')
            threshold: Trust threshold for client selection
            learning_rate: Learning rate for dynamic weight adaptation (η)
            use_dynamic_weights: Whether to use ρ-adaptive dynamic coefficients
            probe_data: Public probe dataset for entropy calculation
        """
        self.trust_mode = trust_mode
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.use_dynamic_weights = use_dynamic_weights
        self.probe_data = probe_data
        self.logger = logging.getLogger(__name__)
        
        # Historical data for reputation calculation
        self.client_history = defaultdict(list)
        self.global_update_history = []
        
        # Raw metric histories for correlation analysis
        self.cosine_history = defaultdict(list)
        self.entropy_history = defaultdict(list) 
        self.reputation_history = defaultdict(list)
        self.accuracy_delta_history = defaultdict(list)
        
        # Dynamic coefficients (θ = [α, β, γ])
        self.theta = np.array([0.4, 0.3, 0.3])  # Initial weights
        self.theta_history = [self.theta.copy()]
        
        # Static weights for backward compatibility
        self.weights = {
            'cosine': self.theta[0],
            'entropy': self.theta[1],
            'reputation': self.theta[2]
        }
    
    def evaluate_trust(self, client_id: str, model_update: Dict[str, torch.Tensor],
                      performance_metrics: Dict[str, float], 
                      global_model: Dict[str, torch.Tensor],
                      round_number: int,
                      global_update_avg: Optional[Dict[str, torch.Tensor]] = None,
                      client_model: Optional[torch.nn.Module] = None,
                      participation_rate: float = 1.0,
                      flags: int = 0) -> float:
        """
        Evaluate trust score for a client based on their model update.
        
        Args:
            client_id: Unique identifier for the client
            model_update: Client's model parameter updates
            performance_metrics: Client's performance metrics (accuracy, loss, etc.)
            global_model: Current global model parameters
            round_number: Current federated learning round
            global_update_avg: Average of all client updates for cosine calculation
            client_model: Client's model for entropy calculation on probe data
            participation_rate: Client's participation rate in recent rounds
            flags: Number of anomaly flags for this client
            
        Returns:
            Trust score between 0 and 1
        """
        if self.trust_mode == 'cosine':
            return self._cosine_trust(model_update, global_model, global_update_avg)
        elif self.trust_mode == 'entropy':
            return self._entropy_trust(model_update, client_model)
        elif self.trust_mode == 'reputation':
            return self._reputation_trust(client_id, performance_metrics, round_number, 
                                        participation_rate, flags)
        elif self.trust_mode == 'hybrid':
            return self._hybrid_trust(client_id, model_update, performance_metrics, 
                                    global_model, round_number, global_update_avg,
                                    client_model, participation_rate, flags)
        else:
            raise ValueError(f"Unknown trust mode: {self.trust_mode}")
    
    def _cosine_trust(self, model_update: Dict[str, torch.Tensor], 
                     global_model: Dict[str, torch.Tensor],
                     global_update_avg: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Calculate trust based on cosine similarity between client update and average global update.
        Implements: cos_i^t = cos(Δw_i^t, Δw̄^t)
        
        Args:
            model_update: Client's model parameter updates (Δw_i^t)
            global_model: Global model parameters (for computing deltas)
            global_update_avg: Average of all client updates (Δw̄^t)
            
        Returns:
            Cosine similarity-based trust score
        """
        # Calculate client update delta (Δw_i^t)
        if len(self.global_update_history) > 0:
            prev_global = self.global_update_history[-1]
            client_delta = {}
            for key in model_update.keys():
                if key in prev_global:
                    client_delta[key] = model_update[key] - prev_global[key]
                else:
                    client_delta[key] = model_update[key]
        else:
            # First round, use the update as delta
            client_delta = model_update
        
        # Use provided global average update or compute from current updates
        if global_update_avg is None:
            # If no global average provided, use the global model as reference
            global_delta = global_model
        else:
            global_delta = global_update_avg
        
        # Flatten parameter deltas
        client_params = torch.cat([param.flatten() for param in client_delta.values()])
        global_params = torch.cat([param.flatten() for param in global_delta.values()])
        
        # Calculate cosine similarity: cos(Δw_i^t, Δw̄^t)
        cosine_sim = F.cosine_similarity(client_params.unsqueeze(0), 
                                       global_params.unsqueeze(0))
        
        # Convert to trust score (already in [-1, 1], normalize to [0, 1])
        trust_score = (cosine_sim.item() + 1) / 2
        
        return max(0.0, min(1.0, trust_score))
    
    def _entropy_trust(self, model_update: Dict[str, torch.Tensor], 
                      client_model: Optional[torch.nn.Module] = None) -> float:
        """
        Calculate trust based on entropy of predictions on a public probe set.
        Implements: ent_i^t = E_x[-∑ p̂_i log p̂_i] on a public probe set
        
        Args:
            model_update: Client's model parameters
            client_model: Client's model for inference (if available)
            
        Returns:
            Entropy-based trust score
        """
        if self.probe_data is not None and client_model is not None:
            # Use public probe set for entropy calculation
            entropies = []
            client_model.eval()
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(self.probe_data):
                    if batch_idx >= 5:  # Limit to first 5 batches for efficiency
                        break
                    
                    # Get predictions from client model
                    outputs = client_model(data)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Calculate entropy for each sample
                    sample_entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    entropies.extend(sample_entropies.cpu().numpy())
            
            if entropies:
                # Average entropy across probe samples
                avg_entropy = np.mean(entropies)
                # Normalize to [0, 1] (higher entropy = higher trust)
                trust_score = min(1.0, avg_entropy / 5.0)  # Assuming max entropy ~5.0
                return max(0.0, trust_score)
        
        # Fallback: Use parameter distribution entropy
        entropies = []
        
        for param_name, param_tensor in model_update.items():
            # Convert to numpy and flatten
            param_flat = param_tensor.detach().cpu().numpy().flatten()
            
            # Create histogram for entropy calculation
            hist, _ = np.histogram(param_flat, bins=50, density=True)
            
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            
            # Calculate entropy
            param_entropy = entropy(hist)
            entropies.append(param_entropy)
        
        # Average entropy across all parameters
        avg_entropy = np.mean(entropies)
        
        # Normalize entropy to [0, 1] range (higher entropy = higher trust)
        trust_score = min(1.0, avg_entropy / 4.0)
        
        return max(0.0, trust_score)
    
    def _reputation_trust(self, client_id: str, performance_metrics: Dict[str, float],
                         round_number: int, participation_rate: float = 1.0, 
                         flags: int = 0) -> float:
        """
        Calculate trust based on historical performance using EMA.
        Implements: rep_i^t = EMA(ΔAcc_i, participation, flags)
        
        Args:
            client_id: Client identifier
            performance_metrics: Current round performance metrics
            round_number: Current round number
            participation_rate: Client's participation rate in recent rounds
            flags: Number of anomaly flags for this client
            
        Returns:
            Reputation-based trust score
        """
        # Store current performance
        current_accuracy = performance_metrics.get('accuracy', 0.0)
        
        # Calculate accuracy delta (ΔAcc_i)
        if client_id in self.client_history and self.client_history[client_id]:
            prev_accuracy = self.client_history[client_id][-1]['accuracy']
            accuracy_delta = current_accuracy - prev_accuracy
        else:
            accuracy_delta = current_accuracy  # First round
        
        # Store current performance
        self.client_history[client_id].append({
            'round': round_number,
            'accuracy': current_accuracy,
            'accuracy_delta': accuracy_delta,
            'loss': performance_metrics.get('loss', 1.0),
            'f1_score': performance_metrics.get('f1_score', 0.0),
            'participation': participation_rate,
            'flags': flags
        })
        
        # Store accuracy delta for correlation analysis
        self.accuracy_delta_history[client_id].append(accuracy_delta)
        
        history = self.client_history[client_id]
        
        if len(history) < 2:
            # Not enough history, use current performance with participation penalty
            base_score = current_accuracy * participation_rate
            flag_penalty = min(0.1 * flags, 0.5)  # Max 50% penalty
            return max(0.0, min(1.0, base_score - flag_penalty))
        
        # EMA calculation for accuracy deltas
        alpha = 0.3  # EMA smoothing factor
        ema_acc_delta = accuracy_delta
        
        for i in range(len(history) - 2, -1, -1):
            prev_delta = history[i]['accuracy_delta']
            ema_acc_delta = alpha * prev_delta + (1 - alpha) * ema_acc_delta
        
        # Normalize EMA to [0, 1] range
        # Assume accuracy deltas typically range from -0.2 to +0.2
        normalized_ema = (ema_acc_delta + 0.2) / 0.4
        normalized_ema = max(0.0, min(1.0, normalized_ema))
        
        # Apply participation rate multiplier
        participation_score = normalized_ema * participation_rate
        
        # Apply flag penalty
        flag_penalty = min(0.1 * flags, 0.5)  # Max 50% penalty
        
        # Final reputation score
        reputation_score = participation_score - flag_penalty
        
        return max(0.0, min(1.0, reputation_score))
    
    def _update_dynamic_weights(self, client_id: str) -> None:
        """
        Update dynamic coefficients using ρ-adaptive method.
        Implements:
        ρ = spearman([cos, ent, rep], ΔAcc) # three correlations
        θ = softplus(θ_prev + η·ρ) # θ = [α,β,γ]
        θ = θ / θ.sum() # simplex projection
        """
        if not self.use_dynamic_weights:
            return
            
        # Need sufficient history for correlation analysis
        min_history = 5
        if (len(self.cosine_history[client_id]) < min_history or
            len(self.entropy_history[client_id]) < min_history or
            len(self.reputation_history[client_id]) < min_history or
            len(self.accuracy_delta_history[client_id]) < min_history):
            return
        
        try:
            # Get recent history for correlation analysis
            recent_window = min(20, len(self.cosine_history[client_id]))
            
            cos_scores = self.cosine_history[client_id][-recent_window:]
            ent_scores = self.entropy_history[client_id][-recent_window:]
            rep_scores = self.reputation_history[client_id][-recent_window:]
            acc_deltas = self.accuracy_delta_history[client_id][-recent_window:]
            
            # Calculate Spearman correlations with accuracy delta
            rho_cos, _ = spearmanr(cos_scores, acc_deltas)
            rho_ent, _ = spearmanr(ent_scores, acc_deltas)
            rho_rep, _ = spearmanr(rep_scores, acc_deltas)
            
            # Handle NaN correlations (replace with 0)
            rho_cos = 0.0 if np.isnan(rho_cos) else rho_cos
            rho_ent = 0.0 if np.isnan(rho_ent) else rho_ent
            rho_rep = 0.0 if np.isnan(rho_rep) else rho_rep
            
            # Create correlation vector ρ
            rho = np.array([rho_cos, rho_ent, rho_rep])
            
            # Update weights: θ = softplus(θ_prev + η·ρ)
            theta_new = self.theta + self.learning_rate * rho
            
            # Apply softplus for positivity
            theta_new = np.log(1 + np.exp(theta_new))
            
            # Simplex projection (normalize to sum to 1)
            theta_new = theta_new / theta_new.sum()
            
            # Update theta
            self.theta = theta_new
            self.theta_history.append(self.theta.copy())
            
            # Update weights dictionary for backward compatibility
            self.weights = {
                'cosine': self.theta[0],
                'entropy': self.theta[1], 
                'reputation': self.theta[2]
            }
            
            self.logger.debug(f"Updated dynamic weights for client {client_id}: "
                            f"cos={self.theta[0]:.3f}, ent={self.theta[1]:.3f}, "
                            f"rep={self.theta[2]:.3f}, correlations=[{rho_cos:.3f}, "
                            f"{rho_ent:.3f}, {rho_rep:.3f}]")
                            
        except Exception as e:
            self.logger.warning(f"Failed to update dynamic weights: {e}")

    def _hybrid_trust(self, client_id: str, model_update: Dict[str, torch.Tensor],
                     performance_metrics: Dict[str, float], 
                     global_model: Dict[str, torch.Tensor],
                     round_number: int, global_update_avg: Optional[Dict[str, torch.Tensor]] = None,
                     client_model: Optional[torch.nn.Module] = None,
                     participation_rate: float = 1.0, flags: int = 0) -> float:
        """
        Calculate hybrid trust combining multiple trust metrics with dynamic weights.
        
        Args:
            client_id: Client identifier
            model_update: Client's model parameters
            performance_metrics: Current round performance metrics
            global_model: Global model parameters
            round_number: Current round number
            global_update_avg: Average of all client updates for cosine calculation
            client_model: Client's model for entropy calculation
            participation_rate: Client's participation rate
            flags: Number of anomaly flags
            
        Returns:
            Hybrid trust score using dynamic or static weights
        """
        # Calculate individual trust components
        cosine_trust = self._cosine_trust(model_update, global_model, global_update_avg)
        entropy_trust = self._entropy_trust(model_update, client_model)
        reputation_trust = self._reputation_trust(client_id, performance_metrics, 
                                                round_number, participation_rate, flags)
        
        # Store metrics for correlation analysis
        self.cosine_history[client_id].append(cosine_trust)
        self.entropy_history[client_id].append(entropy_trust)
        self.reputation_history[client_id].append(reputation_trust)
        
        # Update dynamic weights based on correlations
        self._update_dynamic_weights(client_id)
        
        # Calculate hybrid score using current weights
        hybrid_score = (self.weights['cosine'] * cosine_trust +
                       self.weights['entropy'] * entropy_trust +
                       self.weights['reputation'] * reputation_trust)
        
        # Log individual components for debugging
        self.logger.debug(f"Client {client_id} trust components - "
                         f"Cosine: {cosine_trust:.3f}, "
                         f"Entropy: {entropy_trust:.3f}, "
                         f"Reputation: {reputation_trust:.3f}, "
                         f"Weights: [{self.weights['cosine']:.3f}, "
                         f"{self.weights['entropy']:.3f}, {self.weights['reputation']:.3f}], "
                         f"Hybrid: {hybrid_score:.3f}")
        
        return max(0.0, min(1.0, hybrid_score))
    
    def select_trusted_clients(self, available_clients: List[str],
                              client_trust_scores: Dict[str, List[float]],
                              selection_ratio: float = 0.8) -> List[str]:
        """
        Select trusted clients based on their trust scores.
        Drop clients with trust < τ, re-weight the rest by trust/Σtrust.
        
        Args:
            available_clients: List of available client IDs
            client_trust_scores: Historical trust scores for each client
            selection_ratio: Fraction of clients to select
            
        Returns:
            List of selected trusted client IDs
        """
        # Get latest trust scores for available clients
        client_scores = {}
        for client_id in available_clients:
            if client_id in client_trust_scores and client_trust_scores[client_id]:
                client_scores[client_id] = client_trust_scores[client_id][-1]
            else:
                # New client, assign neutral trust
                client_scores[client_id] = 0.5
        
        # Filter clients above threshold (trust < τ)
        trusted_clients = [client_id for client_id, score in client_scores.items() 
                          if score >= self.threshold]
        
        if not trusted_clients:
            # If no clients meet threshold, select top performers
            self.logger.warning(f"No clients meet trust threshold {self.threshold}. "
                              f"Using top {max(1, int(len(available_clients) * 0.5))} performers.")
            trusted_clients = sorted(client_scores.keys(), 
                                   key=lambda x: client_scores[x], 
                                   reverse=True)[:max(1, int(len(available_clients) * 0.5))]
        
        # Calculate sum of trust scores for normalization
        trusted_scores = {client_id: client_scores[client_id] for client_id in trusted_clients}
        sum_trust = sum(trusted_scores.values())
        
        # Normalize trust scores (trust / Σtrust)
        if sum_trust > 0:
            normalized_scores = {client_id: score / sum_trust 
                               for client_id, score in trusted_scores.items()}
        else:
            # Equal weighting if all scores are 0
            normalized_scores = {client_id: 1.0 / len(trusted_clients) 
                               for client_id in trusted_clients}
        
        # Select based on ratio
        num_selected = max(1, int(len(trusted_clients) * selection_ratio))
        
        # Probability-based selection weighted by normalized trust scores
        if len(trusted_clients) <= num_selected:
            selected = trusted_clients
        else:
            # Convert to arrays for weighted selection
            client_ids = list(normalized_scores.keys())
            probabilities = np.array(list(normalized_scores.values()))
            
            # Ensure probabilities sum to 1
            probabilities = probabilities / probabilities.sum()
            
            selected = np.random.choice(
                client_ids,
                size=num_selected,
                replace=False,
                p=probabilities
            ).tolist()
        
        self.logger.info(f"Selected {len(selected)} trusted clients out of {len(available_clients)} available")
        return selected
    
    def detect_malicious_clients(self, client_trust_scores: Dict[str, List[float]],
                                detection_window: int = 5,
                                malicious_threshold: float = 0.3) -> List[str]:
        """
        Detect potentially malicious clients based on trust score patterns.
        
        Args:
            client_trust_scores: Historical trust scores for each client
            detection_window: Number of recent rounds to consider
            malicious_threshold: Threshold below which client is considered malicious
            
        Returns:
            List of potentially malicious client IDs
        """
        malicious_clients = []
        
        for client_id, scores in client_trust_scores.items():
            if len(scores) >= detection_window:
                recent_scores = scores[-detection_window:]
                avg_recent_trust = np.mean(recent_scores)
                
                # Check for consistently low trust
                if avg_recent_trust < malicious_threshold:
                    malicious_clients.append(client_id)
                
                # Check for sudden drop in trust
                elif len(scores) > detection_window:
                    previous_avg = np.mean(scores[-(detection_window*2):-detection_window])
                    if previous_avg - avg_recent_trust > 0.4:  # Significant drop
                        malicious_clients.append(client_id)
        
        if malicious_clients:
            self.logger.warning(f"Detected potentially malicious clients: {malicious_clients}")
        
        return malicious_clients
    
    def update_trust_weights(self, cosine_weight: float, entropy_weight: float, 
                           reputation_weight: float):
        """
        Update weights for hybrid trust calculation.
        
        Args:
            cosine_weight: Weight for cosine similarity component
            entropy_weight: Weight for entropy component
            reputation_weight: Weight for reputation component
        """
        total = cosine_weight + entropy_weight + reputation_weight
        self.weights = {
            'cosine': cosine_weight / total,
            'entropy': entropy_weight / total,
            'reputation': reputation_weight / total
        }
        self.logger.info(f"Updated trust weights: {self.weights}")
    
    def get_trust_statistics(self, client_trust_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Get comprehensive trust statistics across all clients.
        
        Args:
            client_trust_scores: Historical trust scores for each client
            
        Returns:
            Dictionary containing trust statistics
        """
        if not client_trust_scores:
            return {}
        
        all_scores = []
        for scores in client_trust_scores.values():
            all_scores.extend(scores)
        
        if not all_scores:
            return {}
        
        stats = {
            'mean_trust': np.mean(all_scores),
            'std_trust': np.std(all_scores),
            'min_trust': np.min(all_scores),
            'max_trust': np.max(all_scores),
            'num_clients': len(client_trust_scores),
            'total_evaluations': len(all_scores)
        }
        
        # Client-specific statistics
        client_stats = {}
        for client_id, scores in client_trust_scores.items():
            if scores:
                client_stats[client_id] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'trend': scores[-1] - scores[0] if len(scores) > 1 else 0.0,
                    'evaluations': len(scores)
                }
        
        stats['client_statistics'] = client_stats
        return stats
    
    def aggregate_model_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                            client_trust_scores: Dict[str, float],
                            trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using trust-weighted trimmed mean.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            client_trust_scores: Dictionary mapping client IDs to their trust scores
            trim_ratio: Ratio of extreme values to trim (from each end)
            
        Returns:
            Aggregated model update
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Filter clients based on trust threshold
        trusted_clients = [client_id for client_id, trust in client_trust_scores.items() 
                          if trust >= self.threshold]
        
        if not trusted_clients:
            self.logger.warning("No clients meet trust threshold for aggregation. "
                              "Using all clients with re-weighted trust scores.")
            trusted_clients = list(client_trust_scores.keys())
        
        # Get updates from trusted clients
        trusted_updates = {client_id: client_updates[client_id] 
                          for client_id in trusted_clients if client_id in client_updates}
        
        if not trusted_updates:
            raise ValueError("No trusted client updates available for aggregation")
        
        # Get trust scores for trusted clients
        trust_scores = {client_id: client_trust_scores[client_id] for client_id in trusted_updates}
        
        # Normalize trust scores to sum to 1
        sum_trust = sum(trust_scores.values())
        if sum_trust > 0:
            normalized_weights = {client_id: score / sum_trust 
                                for client_id, score in trust_scores.items()}
        else:
            # Equal weighting if all scores are 0
            normalized_weights = {client_id: 1.0 / len(trusted_updates) 
                                for client_id in trusted_updates}
        
        # Initialize aggregated model with zeros
        first_client_id = list(trusted_updates.keys())[0]
        aggregated_model = {}
        
        # For each parameter in the model
        for param_name in trusted_updates[first_client_id].keys():
            # Get weighted parameter updates from all trusted clients
            weighted_params = []
            weights = []
            
            for client_id, update in trusted_updates.items():
                if param_name in update:
                    weighted_params.append(update[param_name])
                    weights.append(normalized_weights[client_id])
            
            if not weighted_params:
                continue
                
            # Stack tensors for trimmed mean calculation
            stacked_params = torch.stack(weighted_params, dim=0)
            weight_tensor = torch.tensor(weights, device=stacked_params.device).view(-1, 1, 1, 1)
            
            # Calculate trimmed mean for robust aggregation
            if len(weighted_params) >= 4:  # Need sufficient samples for trimming
                k = max(1, int(trim_ratio * len(weighted_params)))
                sorted_indices = torch.argsort(stacked_params, dim=0)
                
                # Remove k smallest and k largest values
                trimmed_indices = sorted_indices[k:-k]
                trimmed_params = torch.gather(stacked_params, 0, trimmed_indices)
                
                # Calculate weighted mean of trimmed parameters
                trimmed_weights = torch.gather(weight_tensor, 0, 
                                            trimmed_indices.unsqueeze(1).expand_as(weight_tensor))
                trimmed_weights = trimmed_weights / trimmed_weights.sum()
                
                # Calculate weighted trimmed mean
                aggregated_model[param_name] = torch.sum(trimmed_params * trimmed_weights, dim=0)
            else:
                # Use weighted mean if not enough samples for trimming
                aggregated_model[param_name] = torch.sum(stacked_params * weight_tensor, dim=0)
        
        self.logger.info(f"Aggregated model updates from {len(trusted_updates)} trusted clients "
                       f"using trust-weighted trimmed mean (trim_ratio={trim_ratio})")
        
        return aggregated_model
    
    def compute_update_average(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute the average of all client model updates.
        This is used for cosine similarity calculations: Δw̄^t.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            
        Returns:
            Average model update across all clients
        """
        if not client_updates:
            raise ValueError("No client updates provided for averaging")
        
        # Initialize with zeros from the first client
        first_client_id = list(client_updates.keys())[0]
        update_avg = {}
        
        # For each parameter in the model
        for param_name in client_updates[first_client_id].keys():
            # Collect parameters from all clients
            param_tensors = []
            
            for client_id, update in client_updates.items():
                if param_name in update:
                    param_tensors.append(update[param_name])
            
            if not param_tensors:
                continue
                
            # Stack and average
            stacked_params = torch.stack(param_tensors, dim=0)
            update_avg[param_name] = torch.mean(stacked_params, dim=0)
        
        return update_avg
    
    def update_global_model_history(self, global_model: Dict[str, torch.Tensor]) -> None:
        """
        Update the history of global models.
        This is used for calculating client update deltas: Δw_i^t.
        
        Args:
            global_model: Current global model parameters
        """
        # Store a copy of the global model
        global_copy = {k: v.clone().detach() for k, v in global_model.items()}
        
        # Append to history
        self.global_update_history.append(global_copy)
        
        # Keep only recent history to manage memory
        max_history = 5
        if len(self.global_update_history) > max_history:
            self.global_update_history.pop(0)
        
        self.logger.debug(f"Updated global model history. History size: {len(self.global_update_history)}")
        
    def get_dynamic_weight_history(self) -> List[np.ndarray]:
        """
        Get the history of dynamic weight changes.
        Useful for monitoring trust component importance over time.
        
        Returns:
            List of weight vectors [α, β, γ] over time
        """
        return self.theta_history
