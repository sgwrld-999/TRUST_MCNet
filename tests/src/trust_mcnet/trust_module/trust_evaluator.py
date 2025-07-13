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
        try:
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
                # First round, treat the model update itself as delta
                client_delta = model_update
            
            # Use provided global average update (Δw̄^t) or fallback to global model
            if global_update_avg is not None and len(global_update_avg) > 0:
                global_delta = global_update_avg
            else:
                # Fallback: use global model as reference
                if len(self.global_update_history) > 0:
                    prev_global = self.global_update_history[-1]
                    global_delta = {}
                    for key in global_model.keys():
                        if key in prev_global:
                            global_delta[key] = global_model[key] - prev_global[key]
                        else:
                            global_delta[key] = global_model[key]
                else:
                    global_delta = global_model
            
            # Flatten parameter deltas for cosine similarity calculation
            client_params_list = []
            global_params_list = []
            
            for key in client_delta.keys():
                if key in global_delta:
                    client_params_list.append(client_delta[key].flatten())
                    global_params_list.append(global_delta[key].flatten())
            
            if not client_params_list:
                self.logger.warning("No matching parameters for cosine similarity calculation")
                return 0.5  # Neutral trust score
            
            # Concatenate all parameter deltas
            client_params = torch.cat(client_params_list, dim=0)
            global_params = torch.cat(global_params_list, dim=0)
            
            # Handle edge cases
            if torch.norm(client_params) == 0 or torch.norm(global_params) == 0:
                self.logger.debug("Zero norm detected in cosine similarity calculation")
                return 0.5  # Neutral trust for zero updates
            
            # Calculate cosine similarity: cos(Δw_i^t, Δw̄^t)
            cosine_sim = F.cosine_similarity(client_params.unsqueeze(0), 
                                           global_params.unsqueeze(0), dim=1)
            
            # Convert from [-1, 1] to [0, 1] range
            trust_score = (cosine_sim.item() + 1) / 2
            
            # Ensure bounds
            trust_score = max(0.0, min(1.0, trust_score))
            
            self.logger.debug(f"Cosine similarity: {cosine_sim.item():.4f}, "
                            f"Trust score: {trust_score:.4f}")
            
            return trust_score
            
        except Exception as e:
            self.logger.warning(f"Cosine trust calculation failed: {e}")
            return 0.5  # Neutral trust on error
    
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
        try:
            if self.probe_data is not None and client_model is not None:
                # Use public probe set for entropy calculation (preferred method)
                entropies = []
                client_model.eval()
                
                with torch.no_grad():
                    for batch_idx, (data, _) in enumerate(self.probe_data):
                        if batch_idx >= 10:  # Use more batches for better estimation
                            break
                        
                        try:
                            # Get predictions from client model: p̂_i
                            outputs = client_model(data)
                            
                            # Apply softmax to get probability distribution
                            probs = F.softmax(outputs, dim=1)
                            
                            # Calculate entropy for each sample: -∑ p̂_i log p̂_i
                            # Add small epsilon for numerical stability
                            epsilon = 1e-10
                            sample_entropies = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
                            entropies.extend(sample_entropies.cpu().numpy())
                            
                        except Exception as e:
                            self.logger.debug(f"Error processing batch {batch_idx}: {e}")
                            continue
                
                if entropies:
                    # Calculate expected entropy: E_x[-∑ p̂_i log p̂_i]
                    expected_entropy = np.mean(entropies)
                    
                    # Normalize to [0, 1] range
                    # Higher entropy indicates more uncertainty/diversity (can be good or bad)
                    # We'll use a sigmoid-like transformation for gradual trust mapping
                    max_entropy = np.log(10)  # Assume max 10 classes
                    normalized_entropy = expected_entropy / max_entropy
                    
                    # Transform to trust score: moderate entropy = higher trust
                    # Peak trust at entropy around 50% of maximum
                    optimal_entropy = 0.5
                    entropy_deviation = abs(normalized_entropy - optimal_entropy)
                    trust_score = max(0.0, 1.0 - 2 * entropy_deviation)
                    
                    self.logger.debug(f"Probe entropy: {expected_entropy:.4f}, "
                                    f"normalized: {normalized_entropy:.4f}, "
                                    f"trust: {trust_score:.4f}")
                    
                    return trust_score
                else:
                    self.logger.warning("No valid entropy calculations from probe data")
            
            # Fallback: Use parameter distribution entropy
            entropies = []
            
            for param_name, param_tensor in model_update.items():
                try:
                    # Convert to numpy and flatten
                    param_flat = param_tensor.detach().cpu().numpy().flatten()
                    
                    # Skip if parameter is empty or constant
                    if len(param_flat) == 0 or np.std(param_flat) < 1e-8:
                        continue
                    
                    # Create histogram for entropy calculation with adaptive bins
                    n_bins = min(50, max(10, len(param_flat) // 20))
                    hist, _ = np.histogram(param_flat, bins=n_bins, density=True)
                    
                    # Normalize histogram and add small epsilon
                    hist = hist / (hist.sum() + 1e-10)
                    hist = hist + 1e-10
                    
                    # Calculate entropy: -∑ p log p
                    param_entropy = -np.sum(hist * np.log(hist))
                    entropies.append(param_entropy)
                    
                except Exception as e:
                    self.logger.debug(f"Error calculating entropy for {param_name}: {e}")
                    continue
            
            if not entropies:
                self.logger.warning("No valid parameter entropies calculated")
                return 0.5  # Neutral trust
            
            # Average entropy across all parameters
            avg_entropy = np.mean(entropies)
            
            # Normalize entropy to [0, 1] range
            # Higher entropy indicates more diverse parameters (generally positive)
            max_param_entropy = np.log(50)  # Assume max 50 bins
            trust_score = min(1.0, avg_entropy / max_param_entropy)
            
            # Ensure bounds
            trust_score = max(0.0, min(1.0, trust_score))
            
            self.logger.debug(f"Parameter entropy: {avg_entropy:.4f}, trust: {trust_score:.4f}")
            
            return trust_score
            
        except Exception as e:
            self.logger.warning(f"Entropy trust calculation failed: {e}")
            return 0.5  # Neutral trust on error
    
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
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softplus function: softplus(x) = log(1 + exp(x))
        
        Args:
            x: Input array
            
        Returns:
            Softplus activated array
        """
        # Use numerically stable implementation to avoid overflow
        return np.where(x > 20, x, np.log(1 + np.exp(np.clip(x, -500, 20))))
    
    def _update_dynamic_weights(self, client_id: str) -> None:
        """
        Update dynamic coefficients using ρ-adaptive method.
        Implements the enhanced recommendation:
        ρ = spearman([cos, ent, rep], ΔAcc) # three correlations
        θ = softplus(θ_prev + η·ρ) # θ = [α,β,γ] with numerical stability
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
            # Get recent history for correlation analysis with adaptive window
            max_window = 30
            min_window = 10
            available_history = len(self.cosine_history[client_id])
            recent_window = min(max_window, max(min_window, available_history))
            
            cos_scores = self.cosine_history[client_id][-recent_window:]
            ent_scores = self.entropy_history[client_id][-recent_window:]
            rep_scores = self.reputation_history[client_id][-recent_window:]
            acc_deltas = self.accuracy_delta_history[client_id][-recent_window:]
            
            # Ensure all arrays have the same length
            min_len = min(len(cos_scores), len(ent_scores), len(rep_scores), len(acc_deltas))
            cos_scores = cos_scores[-min_len:]
            ent_scores = ent_scores[-min_len:]
            rep_scores = rep_scores[-min_len:]
            acc_deltas = acc_deltas[-min_len:]
            
            # Calculate Spearman correlations with accuracy delta
            rho_cos, p_cos = spearmanr(cos_scores, acc_deltas)
            rho_ent, p_ent = spearmanr(ent_scores, acc_deltas)
            rho_rep, p_rep = spearmanr(rep_scores, acc_deltas)
            
            # Handle NaN correlations and apply significance weighting
            def process_correlation(rho, p_val):
                if np.isnan(rho) or np.isnan(p_val):
                    return 0.0
                # Weight by significance (lower p-value = higher weight)
                significance_weight = max(0.1, 1.0 - p_val) if p_val <= 1.0 else 0.1
                return rho * significance_weight
            
            rho_cos = process_correlation(rho_cos, p_cos if not np.isnan(p_cos) else 1.0)
            rho_ent = process_correlation(rho_ent, p_ent if not np.isnan(p_ent) else 1.0)
            rho_rep = process_correlation(rho_rep, p_rep if not np.isnan(p_rep) else 1.0)
            
            # Create correlation vector ρ with bounds to prevent extreme updates
            rho = np.array([rho_cos, rho_ent, rho_rep])
            rho = np.clip(rho, -2.0, 2.0)  # Bound correlations for stability
            
            # Adaptive learning rate based on correlation strength
            correlation_strength = np.mean(np.abs(rho))
            adaptive_lr = self.learning_rate * (1.0 + correlation_strength)
            adaptive_lr = min(adaptive_lr, 0.1)  # Cap learning rate
            
            # Update weights: θ = softplus(θ_prev + η·ρ)
            theta_update = self.theta + adaptive_lr * rho
            
            # Apply numerically stable softplus for positivity
            theta_new = self._softplus(theta_update)
            
            # Add small epsilon to prevent zero weights
            epsilon = 1e-6
            theta_new = theta_new + epsilon
            
            # Simplex projection (normalize to sum to 1)
            theta_new = theta_new / theta_new.sum()
            
            # Apply momentum for smoother updates
            momentum = 0.1
            self.theta = momentum * self.theta + (1 - momentum) * theta_new
            self.theta_history.append(self.theta.copy())
            
            # Update weights dictionary for backward compatibility
            self.weights = {
                'cosine': self.theta[0],
                'entropy': self.theta[1], 
                'reputation': self.theta[2]
            }
            
            self.logger.debug(f"Updated dynamic weights for client {client_id}: "
                            f"cos={self.theta[0]:.4f}, ent={self.theta[1]:.4f}, "
                            f"rep={self.theta[2]:.4f}, correlations=[{rho_cos:.4f}, "
                            f"{rho_ent:.4f}, {rho_rep:.4f}], adaptive_lr={adaptive_lr:.4f}")
                            
        except Exception as e:
            self.logger.warning(f"Failed to update dynamic weights for client {client_id}: {e}")
            # Graceful fallback to equal weights
            self.theta = np.array([1/3, 1/3, 1/3])
            self.weights = {'cosine': 1/3, 'entropy': 1/3, 'reputation': 1/3}

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
        Implements the enhanced recommendation:
        Drop clients with trust < τ, re-weight the rest by trust / Σtrust, 
        then apply trimmed-mean to obtain the robust global update.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            client_trust_scores: Dictionary mapping client IDs to their trust scores
            trim_ratio: Ratio of extreme values to trim (from each end)
            
        Returns:
            Aggregated model update using trust-weighted trimmed mean
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Step 1: Drop clients with trust < τ (threshold)
        trusted_clients = [client_id for client_id, trust in client_trust_scores.items() 
                          if trust >= self.threshold]
        
        if not trusted_clients:
            self.logger.warning(f"No clients meet trust threshold {self.threshold}. "
                              f"Using clients with top 50% trust scores.")
            # Use top 50% of clients if none meet threshold
            sorted_clients = sorted(client_trust_scores.items(), key=lambda x: x[1], reverse=True)
            num_keep = max(1, len(sorted_clients) // 2)
            trusted_clients = [client_id for client_id, _ in sorted_clients[:num_keep]]
        
        # Step 2: Get updates from trusted clients
        trusted_updates = {client_id: client_updates[client_id] 
                          for client_id in trusted_clients if client_id in client_updates}
        
        if not trusted_updates:
            raise ValueError("No trusted client updates available for aggregation")
        
        # Step 3: Re-weight the rest by trust / Σtrust
        trust_scores = {client_id: client_trust_scores[client_id] for client_id in trusted_updates}
        sum_trust = sum(trust_scores.values())
        
        if sum_trust > 0:
            normalized_weights = {client_id: score / sum_trust 
                                for client_id, score in trust_scores.items()}
        else:
            # Equal weighting if all scores are 0
            normalized_weights = {client_id: 1.0 / len(trusted_updates) 
                                for client_id in trusted_updates}
        
        # Initialize aggregated model
        first_client_id = list(trusted_updates.keys())[0]
        aggregated_model = {}
        
        # Step 4: Apply trimmed-mean to obtain robust global update
        for param_name in trusted_updates[first_client_id].keys():
            # Collect parameter updates and corresponding weights
            param_updates = []
            weights = []
            
            for client_id, update in trusted_updates.items():
                if param_name in update:
                    param_updates.append(update[param_name])
                    weights.append(normalized_weights[client_id])
            
            if not param_updates:
                continue
            
            # Convert to tensors for processing
            stacked_params = torch.stack(param_updates, dim=0)  # Shape: [num_clients, ...]
            weight_tensor = torch.tensor(weights, device=stacked_params.device, dtype=stacked_params.dtype)
            
            # Apply trimmed mean for robust aggregation
            num_clients = len(param_updates)
            
            if num_clients >= 4:  # Need sufficient samples for trimming
                # Calculate number of values to trim from each end
                k = max(1, int(trim_ratio * num_clients))
                
                # Flatten parameters for easier sorting
                original_shape = stacked_params.shape[1:]
                flattened_params = stacked_params.view(num_clients, -1)
                
                # Sort by parameter values and get median direction for trimming
                param_means = torch.mean(flattened_params, dim=1)
                sorted_indices = torch.argsort(param_means)
                
                # Remove k smallest and k largest updates (by mean parameter value)
                trimmed_indices = sorted_indices[k:-k] if k < num_clients // 2 else sorted_indices
                
                # Get trimmed parameters and weights
                trimmed_params = stacked_params[trimmed_indices]
                trimmed_weights = weight_tensor[trimmed_indices]
                
                # Re-normalize weights after trimming
                trimmed_weights = trimmed_weights / trimmed_weights.sum()
                
                # Calculate trust-weighted mean of trimmed parameters
                # Expand weights to match parameter dimensions
                weight_expanded = trimmed_weights.view(-1, *([1] * len(original_shape)))
                aggregated_param = torch.sum(trimmed_params * weight_expanded, dim=0)
                
                self.logger.debug(f"Parameter {param_name}: Used {len(trimmed_params)} clients "
                                f"after trimming {k} from each end (original: {num_clients})")
            else:
                # Use weighted mean if not enough samples for trimming
                weight_expanded = weight_tensor.view(-1, *([1] * len(stacked_params.shape[1:])))
                aggregated_param = torch.sum(stacked_params * weight_expanded, dim=0)
                
                self.logger.debug(f"Parameter {param_name}: Used weighted mean "
                                f"(insufficient samples for trimming: {num_clients})")
            
            aggregated_model[param_name] = aggregated_param
        
        self.logger.info(f"Aggregated model updates from {len(trusted_updates)} trusted clients "
                       f"(threshold: {self.threshold:.3f}) using trust-weighted trimmed mean "
                       f"(trim_ratio: {trim_ratio:.3f})")
        
        # Log trust distribution for transparency
        trust_values = list(trust_scores.values())
        self.logger.info(f"Trust score distribution - mean: {np.mean(trust_values):.3f}, "
                       f"std: {np.std(trust_values):.3f}, "
                       f"range: [{np.min(trust_values):.3f}, {np.max(trust_values):.3f}]")
        
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
    
    def evaluate_trust_batch(self, client_updates: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate trust scores for a batch of clients.
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            
        Returns:
            Dictionary mapping client IDs to their trust scores
        """
        trust_scores = {}
        
        try:
            # Convert numpy arrays to tensors if needed
            converted_updates = {}
            for client_id, update in client_updates.items():
                if isinstance(update, list) and len(update) > 0:
                    if isinstance(update[0], np.ndarray):
                        # Convert numpy arrays to tensors
                        converted_updates[client_id] = {
                            f'layer_{i}': torch.from_numpy(arr).float() 
                            for i, arr in enumerate(update)
                        }
                    else:
                        # Already tensors or other format
                        converted_updates[client_id] = {
                            f'layer_{i}': update[i] if isinstance(update[i], torch.Tensor) 
                            else torch.tensor(update[i]).float()
                            for i in range(len(update))
                        }
                else:
                    # Empty or invalid update
                    converted_updates[client_id] = {}
            
            # Calculate trust scores for each client
            for client_id, model_update in converted_updates.items():
                try:
                    if len(model_update) == 0:
                        trust_scores[client_id] = 0.0
                        continue
                        
                    # Use simple trust calculation for batch processing
                    if self.trust_mode == 'cosine':
                        trust_score = self._cosine_trust(model_update, {}, {})
                    elif self.trust_mode == 'entropy':
                        trust_score = self._entropy_trust(model_update, None)
                    elif self.trust_mode == 'reputation':
                        trust_score = self._reputation_trust(client_id, {}, 1, 1.0, 0)
                    elif self.trust_mode == 'hybrid':
                        trust_score = self._hybrid_trust(client_id, model_update, {}, {}, 1, {}, None, 1.0, 0)
                    else:
                        trust_score = 0.5  # Default neutral trust
                        
                    trust_scores[client_id] = max(0.0, min(1.0, trust_score))
                    
                except Exception as e:
                    self.logger.warning(f"Trust evaluation failed for client {client_id}: {e}")
                    trust_scores[client_id] = 0.5  # Default neutral trust on error
            
        except Exception as e:
            self.logger.error(f"Batch trust evaluation failed: {e}")
            # Return neutral trust for all clients on error
            trust_scores = {client_id: 0.5 for client_id in client_updates.keys()}
        
        return trust_scores
    
    def get_trust_adaptation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of trust adaptation including dynamic weights evolution.
        
        Returns:
            Dictionary containing trust adaptation insights
        """
        summary = {
            'dynamic_weights_enabled': self.use_dynamic_weights,
            'current_weights': {
                'cosine': float(self.theta[0]),
                'entropy': float(self.theta[1]),
                'reputation': float(self.theta[2])
            },
            'learning_rate': self.learning_rate,
            'trust_threshold': self.threshold,
            'weight_evolution': []
        }
        
        # Add weight evolution history
        if len(self.theta_history) > 1:
            for i, weights in enumerate(self.theta_history):
                summary['weight_evolution'].append({
                    'round': i,
                    'cosine': float(weights[0]),
                    'entropy': float(weights[1]),
                    'reputation': float(weights[2])
                })
        
        # Calculate adaptation statistics
        if len(self.theta_history) > 5:
            recent_weights = np.array(self.theta_history[-5:])
            summary['adaptation_stats'] = {
                'weight_stability': {
                    'cosine_std': float(np.std(recent_weights[:, 0])),
                    'entropy_std': float(np.std(recent_weights[:, 1])),
                    'reputation_std': float(np.std(recent_weights[:, 2]))
                },
                'dominant_metric': ['cosine', 'entropy', 'reputation'][np.argmax(self.theta)],
                'weight_convergence': float(np.mean(np.std(recent_weights, axis=0)))
            }
        
        return summary
    
    def analyze_trust_effectiveness(self, window_size: int = 20) -> Dict[str, Any]:
        """
        Analyze effectiveness of trust metrics in predicting accuracy improvements.
        
        Args:
            window_size: Number of recent rounds to analyze
            
        Returns:
            Analysis of trust metric effectiveness
        """
        analysis = {
            'overall_correlations': {},
            'recommendations': []
        }
        
        # Aggregate correlations across all clients
        all_cosine = []
        all_entropy = []
        all_reputation = []
        all_accuracy_deltas = []
        
        for client_id in self.cosine_history.keys():
            if len(self.cosine_history[client_id]) >= window_size:
                recent_cos = self.cosine_history[client_id][-window_size:]
                recent_ent = self.entropy_history[client_id][-window_size:]
                recent_rep = self.reputation_history[client_id][-window_size:]
                recent_acc = self.accuracy_delta_history[client_id][-window_size:]
                
                all_cosine.extend(recent_cos)
                all_entropy.extend(recent_ent)
                all_reputation.extend(recent_rep)
                all_accuracy_deltas.extend(recent_acc)
        
        # Calculate overall correlations
        if len(all_cosine) > 10:
            try:
                cos_corr, _ = spearmanr(all_cosine, all_accuracy_deltas)
                ent_corr, _ = spearmanr(all_entropy, all_accuracy_deltas)
                rep_corr, _ = spearmanr(all_reputation, all_accuracy_deltas)
                
                analysis['overall_correlations'] = {
                    'cosine': float(cos_corr) if not np.isnan(cos_corr) else 0.0,
                    'entropy': float(ent_corr) if not np.isnan(ent_corr) else 0.0,
                    'reputation': float(rep_corr) if not np.isnan(rep_corr) else 0.0
                }
                
                # Generate recommendations
                correlations = analysis['overall_correlations']
                max_corr_metric = max(correlations.keys(), key=lambda k: abs(correlations[k]))
                
                if abs(correlations[max_corr_metric]) > 0.3:
                    analysis['recommendations'].append(
                        f"'{max_corr_metric}' metric shows strongest correlation with accuracy "
                        f"({correlations[max_corr_metric]:.3f})"
                    )
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate correlations: {e}")
                analysis['overall_correlations'] = {'cosine': 0.0, 'entropy': 0.0, 'reputation': 0.0}
        
        return analysis
