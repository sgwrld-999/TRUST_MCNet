"""
Trust evaluation module for TRUST-MCNet federated learning framework.

This module implements various trust evaluation mechanisms including:
- Cosine similarity-based trust
- Entropy-based trust evaluation  
- Reputation-based trust scoring
- Hybrid trust combination methods
- Quarantine-based client filtering
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Union, Tuple, Optional
from scipy.stats import entropy, spearmanr
from collections import defaultdict
import logging
import warnings

from .quarantine_state import QuarantineState


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
                 probe_data: Optional[torch.utils.data.DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize trust evaluator.
        
        Args:
            trust_mode: Trust evaluation method ('cosine', 'entropy', 'reputation', 'hybrid')
            threshold: Trust threshold for client selection
            learning_rate: Learning rate for dynamic weight adaptation (η)
            use_dynamic_weights: Whether to use ρ-adaptive dynamic coefficients
            probe_data: Public probe dataset for entropy calculation
            config: Configuration dictionary including quarantine parameters
        """
        self.trust_mode = trust_mode
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.use_dynamic_weights = use_dynamic_weights
        self.probe_data = probe_data
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize quarantine state manager
        self.quarantine_state = QuarantineState(self.config)
        
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
    
    def detect_malicious_clients(
        self,
        client_ids: List[str],
        trust_vec: List[float],
        round_number: int = 0
    ) -> Tuple[List[str], List[str]]:
        """
        Enhanced malicious client detection with quarantine logic.
        
        Automatically excludes or heavily down-weights clients whose trust score falls 
        below threshold τ for Q consecutive rounds, implementing the quarantine hook.
        
        Args:
            client_ids: List of client IDs to evaluate
            trust_vec: Corresponding trust scores for each client
            round_number: Current training round number
            
        Returns:
            Tuple of (quarantined_clients, surviving_clients)
        """
        # Get quarantine configuration
        quarantine_config = self.config.get('trust', {}).get('quarantine', {})
        tau = quarantine_config.get('tau', 0.35)
        patience = quarantine_config.get('patience', 2)
        quarantine_rounds = quarantine_config.get('quarantine_rounds', 5)
        enable_quarantine = quarantine_config.get('enable_quarantine', True)
        
        if not enable_quarantine:
            # Quarantine disabled, return all as survivors
            return [], client_ids
        
        quarantined = []
        survivors = []
        
        # Update quarantine state for each client
        for client_id, trust_score in zip(client_ids, trust_vec):
            self.quarantine_state.update_client_status(
                client_id=client_id,
                trust_score=trust_score,
                round_number=round_number,
                tau=tau,
                patience=patience,
                quarantine_rounds=quarantine_rounds
            )
            
            # Check if client is currently quarantined
            if self.quarantine_state.is_quarantined(client_id):
                quarantined.append(client_id)
            else:
                survivors.append(client_id)
        
        # Log quarantine decisions
        if quarantined:
            self.logger.info(f"[Round {round_number}] Quarantined clients: {quarantined}")
        if survivors:
            self.logger.info(f"[Round {round_number}] Surviving clients: {survivors}")
        
        # Log quarantine statistics
        stats = self.quarantine_state.get_quarantine_statistics()
        self.logger.info(f"[Round {round_number}] Quarantine stats: "
                        f"{stats['currently_quarantined']}/{stats['total_clients']} quarantined "
                        f"({stats['quarantine_rate']:.1%})")
        
        return quarantined, survivors
    
    def get_quarantine_statistics(self) -> Dict[str, Any]:
        """
        Get current quarantine statistics for monitoring.
        
        Returns:
            Dictionary containing quarantine statistics
        """
        return self.quarantine_state.get_quarantine_statistics()
    
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
    
    def aggregate_model_updates(
        self, 
        client_updates: Dict[str, Dict[str, torch.Tensor]], 
        client_trust_scores: Dict[str, float],
        round_number: int = 0,
        trim_ratio: Optional[float] = None,
        metrics_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Enhanced aggregate client model updates with quarantine logic, trust-weighted trimmed mean,
        and SHAP-aligned trust attribution.
        
        Implements the complete quarantine/trimming logic hook:
        1. Detect and quarantine clients with sustained low trust
        2. Compute SHAP alignment scores if fingerprints provided
        3. Apply trust-weighted trimmed mean on surviving clients
        4. Return aggregated model with comprehensive trust statistics
        
        Args:
            client_updates: Dictionary mapping client IDs to their model updates
            client_trust_scores: Dictionary mapping client IDs to their trust scores
            round_number: Current federated learning round number
            trim_ratio: Ratio of extreme values to trim (from config if None)
            metrics_list: List of metrics dictionaries with SHAP fingerprints (optional)
                         Expected format: [{"client_id": str, "shap": List[float], ...}, ...]
            
        Returns:
            Tuple of (aggregated_model, trust_statistics)
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Get configuration parameters
        aggregation_config = self.config.get('trust', {}).get('aggregation', {})
        trust_config = self.config.get('trust', {})
        
        if trim_ratio is None:
            trim_ratio = aggregation_config.get('trim_ratio', 0.2)
        
        # Step 1: Process SHAP fingerprints and compute alignment scores
        shap_alignment_scores = {}
        if metrics_list is not None:
            shap_alignment_scores = self._compute_shap_alignment_scores(metrics_list, round_number)
            self.logger.info(f"[Round {round_number}] Computed SHAP alignment for {len(shap_alignment_scores)} clients")
        
        # Step 2: Enhance trust scores with SHAP alignment (if available)
        enhanced_trust_scores = client_trust_scores.copy()
        if shap_alignment_scores:
            enhanced_trust_scores = self._integrate_shap_with_trust(
                client_trust_scores, shap_alignment_scores, trust_config
            )
            self.logger.info(f"[Round {round_number}] Enhanced trust scores with SHAP alignment")
        
        # Step 3: Apply quarantine logic to detect and exclude sustained low-trust clients
        client_ids = list(enhanced_trust_scores.keys())
        trust_vec = [enhanced_trust_scores[cid] for cid in client_ids]
        
        quarantined_clients, surviving_clients = self.detect_malicious_clients(
            client_ids=client_ids,
            trust_vec=trust_vec,
            round_number=round_number
        )
        
        # Step 4: Filter client updates by survivors (quarantine filtering)
        surviving_updates = {
            client_id: client_updates[client_id] 
            for client_id in surviving_clients 
            if client_id in client_updates
        }
        
        if not surviving_updates:
            # Fallback: if all clients quarantined, use best available client
            best_client = max(enhanced_trust_scores.items(), key=lambda x: x[1])
            surviving_updates = {best_client[0]: client_updates[best_client[0]]}
            surviving_clients = [best_client[0]]
            self.logger.warning(f"All clients quarantined! Using best client: {best_client[0]} "
                              f"(trust: {best_client[1]:.3f})")
        
        # Step 5: Apply traditional trust threshold filtering on survivors
        surviving_trust_scores = {cid: enhanced_trust_scores[cid] for cid in surviving_clients}
        trusted_survivors = [
            client_id for client_id, trust in surviving_trust_scores.items() 
            if trust >= self.threshold
        ]
        
        if not trusted_survivors:
            # Use top 50% of survivors if none meet threshold
            sorted_survivors = sorted(surviving_trust_scores.items(), key=lambda x: x[1], reverse=True)
            num_keep = max(1, len(sorted_survivors) // 2)
            trusted_survivors = [client_id for client_id, _ in sorted_survivors[:num_keep]]
            self.logger.warning(f"No survivors meet trust threshold {self.threshold}. "
                              f"Using top {num_keep} survivors.")
        
        # Step 6: Get final trusted updates
        final_trusted_updates = {
            client_id: surviving_updates[client_id] 
            for client_id in trusted_survivors 
            if client_id in surviving_updates
        }
        
        # Step 7: Re-weight by normalized trust scores
        final_trust_scores = {cid: surviving_trust_scores[cid] for cid in final_trusted_updates}
        sum_trust = sum(final_trust_scores.values())
        
        if sum_trust > 0:
            normalized_weights = {
                client_id: score / sum_trust 
                for client_id, score in final_trust_scores.items()
            }
        else:
            # Equal weighting fallback
            normalized_weights = {
                client_id: 1.0 / len(final_trusted_updates) 
                for client_id in final_trusted_updates
            }
        
        # Step 8: Apply trust-weighted trimmed mean aggregation
        aggregated_model = self._apply_trimmed_mean_aggregation(
            final_trusted_updates, normalized_weights, trim_ratio
        )
        
        # Step 9: Update global reference fingerprint if SHAP was used
        if shap_alignment_scores:
            self._update_global_reference_fingerprint(metrics_list, trusted_survivors)
        
        # Step 10: Compile comprehensive trust statistics
        trust_statistics = {
            'round_number': round_number,
            'total_clients': len(client_ids),
            'quarantined_clients': quarantined_clients,
            'surviving_clients': surviving_clients,
            'trusted_survivors': trusted_survivors,
            'num_quarantined': len(quarantined_clients),
            'num_survivors': len(surviving_clients),
            'num_final_trusted': len(final_trusted_updates),
            'quarantine_rate': len(quarantined_clients) / len(client_ids) if client_ids else 0,
            'trust_threshold': self.threshold,
            'trim_ratio': trim_ratio,
            'aggregation_weights': normalized_weights,
            'trust_scores_distribution': {
                'mean': np.mean(trust_vec),
                'std': np.std(trust_vec),
                'min': np.min(trust_vec),
                'max': np.max(trust_vec)
            },
            'quarantine_stats': self.quarantine_state.get_quarantine_statistics(),
            'shap_enabled': len(shap_alignment_scores) > 0,
            'shap_alignment_scores': shap_alignment_scores,
            'original_trust_scores': client_trust_scores,
            'enhanced_trust_scores': enhanced_trust_scores
        }
        
        # Log aggregation summary
        shap_info = f" | SHAP: {len(shap_alignment_scores)} clients" if shap_alignment_scores else ""
        self.logger.info(
            f"[Round {round_number}] Aggregation complete: "
            f"{len(final_trusted_updates)}/{len(client_ids)} clients used "
            f"({len(quarantined_clients)} quarantined, {len(surviving_clients)} survived, "
            f"{len(final_trusted_updates)} trusted) | "
            f"Trim ratio: {trim_ratio:.2f} | "
            f"Trust range: [{np.min(trust_vec):.3f}, {np.max(trust_vec):.3f}]"
            f"{shap_info}"
        )
        
        return aggregated_model, trust_statistics
    
    def _apply_trimmed_mean_aggregation(
        self,
        trusted_updates: Dict[str, Dict[str, torch.Tensor]],
        normalized_weights: Dict[str, float],
        trim_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply trust-weighted trimmed mean aggregation to client updates.
        
        Args:
            trusted_updates: Dictionary of client updates to aggregate
            normalized_weights: Normalized trust weights for each client
            trim_ratio: Ratio of extreme values to trim from each end
            
        Returns:
            Aggregated model parameters
        """
        if not trusted_updates:
            raise ValueError("No trusted updates to aggregate")
        
        # Initialize aggregated model
        first_client_id = list(trusted_updates.keys())[0]
        aggregated_model = {}
        
        # Get minimum clients for trimming from config
        aggregation_config = self.config.get('trust', {}).get('aggregation', {})
        min_clients_for_trimming = aggregation_config.get('min_clients_for_trimming', 4)
        
        # Apply trimmed-mean to each parameter
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
            
            if num_clients >= min_clients_for_trimming:
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
                weight_expanded = trimmed_weights.view(-1, *([1] * len(original_shape)))
                aggregated_param = torch.sum(trimmed_params * weight_expanded, dim=0)
                
                self.logger.debug(f"Parameter {param_name}: Trimmed {k} from each end, "
                                f"used {len(trimmed_params)}/{num_clients} clients")
            else:
                # Use weighted mean if not enough samples for trimming
                weight_expanded = weight_tensor.view(-1, *([1] * len(stacked_params.shape[1:])))
                aggregated_param = torch.sum(stacked_params * weight_expanded, dim=0)
                
                self.logger.debug(f"Parameter {param_name}: Used weighted mean "
                                f"(insufficient clients for trimming: {num_clients})")
            
            aggregated_model[param_name] = aggregated_param
        
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
    
    def _compute_shap_alignment_scores(self, metrics_list: List[Dict[str, Any]], round_number: int) -> Dict[str, float]:
        """
        Compute SHAP alignment scores for clients based on their feature attribution fingerprints.
        
        Implements: shap_alignment_i = cos(shap_i, global_ref) with global reference tracking.
        
        Args:
            metrics_list: List of metrics dictionaries containing SHAP fingerprints
                         Expected format: [{"client_id": str, "shap": List[float], ...}, ...]
            round_number: Current federated learning round
            
        Returns:
            Dictionary mapping client IDs to their SHAP alignment scores [0, 1]
        """
        alignment_scores = {}
        
        try:
            # Extract SHAP fingerprints from metrics
            client_fingerprints = {}
            for metrics in metrics_list:
                if isinstance(metrics, dict) and "client_id" in metrics and "shap" in metrics:
                    client_id = metrics["client_id"]
                    shap_fingerprint = metrics["shap"]
                    
                    # Validate fingerprint format
                    if isinstance(shap_fingerprint, list) and len(shap_fingerprint) > 0:
                        # Convert to numpy array and normalize
                        fingerprint_array = np.array(shap_fingerprint, dtype=np.float32)
                        
                        # Handle NaN/inf values
                        if np.any(np.isnan(fingerprint_array)) or np.any(np.isinf(fingerprint_array)):
                            self.logger.warning(f"Invalid SHAP fingerprint for client {client_id}, using zero vector")
                            fingerprint_array = np.zeros_like(fingerprint_array)
                        
                        # L2 normalize for cosine similarity
                        norm = np.linalg.norm(fingerprint_array)
                        if norm > 1e-8:
                            fingerprint_array = fingerprint_array / norm
                        
                        client_fingerprints[client_id] = fingerprint_array
                    else:
                        self.logger.warning(f"Invalid SHAP fingerprint format for client {client_id}")
            
            if not client_fingerprints:
                self.logger.warning("No valid SHAP fingerprints found in metrics_list")
                return alignment_scores
            
            # Initialize or update global reference fingerprint
            global_reference = self._get_or_create_global_reference(client_fingerprints, round_number)
            
            # Compute alignment scores using cosine similarity
            for client_id, fingerprint in client_fingerprints.items():
                try:
                    # Ensure both vectors have the same dimensionality
                    if len(fingerprint) != len(global_reference):
                        self.logger.warning(f"Dimension mismatch for client {client_id}: "
                                          f"{len(fingerprint)} vs {len(global_reference)}")
                        alignment_scores[client_id] = 0.5  # Neutral score
                        continue
                    
                    # Compute cosine similarity with global reference
                    cosine_sim = np.dot(fingerprint, global_reference)
                    
                    # Convert from [-1, 1] to [0, 1] range for alignment score
                    alignment_score = (cosine_sim + 1.0) / 2.0
                    alignment_score = max(0.0, min(1.0, alignment_score))
                    
                    alignment_scores[client_id] = alignment_score
                    
                    self.logger.debug(f"Client {client_id} SHAP alignment: {alignment_score:.4f} "
                                    f"(cosine: {cosine_sim:.4f})")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute SHAP alignment for client {client_id}: {e}")
                    alignment_scores[client_id] = 0.5  # Neutral score on error
            
        except Exception as e:
            self.logger.error(f"SHAP alignment computation failed: {e}")
            # Return neutral scores for all clients on error
            for metrics in metrics_list:
                if isinstance(metrics, dict) and "client_id" in metrics:
                    alignment_scores[metrics["client_id"]] = 0.5
        
        return alignment_scores
    
    def _get_or_create_global_reference(self, client_fingerprints: Dict[str, np.ndarray], 
                                       round_number: int) -> np.ndarray:
        """
        Get or create the global reference fingerprint for SHAP alignment.
        
        Uses exponential moving average to maintain a stable global reference:
        global_ref = α * current_avg + (1-α) * global_ref_prev
        
        Args:
            client_fingerprints: Dictionary of normalized client fingerprints
            round_number: Current federated learning round
            
        Returns:
            Global reference fingerprint (L2 normalized)
        """
        if not hasattr(self, '_global_shap_reference'):
            self._global_shap_reference = None
        
        # Calculate current average fingerprint from all clients
        fingerprint_arrays = list(client_fingerprints.values())
        current_avg = np.mean(fingerprint_arrays, axis=0)
        
        # Initialize global reference on first use
        if self._global_shap_reference is None or round_number <= 1:
            self._global_shap_reference = current_avg.copy()
            self.logger.info(f"Initialized global SHAP reference with {len(current_avg)} features")
        else:
            # Update using exponential moving average
            alpha = 0.1  # EMA smoothing factor (configurable)
            self._global_shap_reference = (alpha * current_avg + 
                                         (1 - alpha) * self._global_shap_reference)
        
        # Ensure global reference is L2 normalized
        norm = np.linalg.norm(self._global_shap_reference)
        if norm > 1e-8:
            self._global_shap_reference = self._global_shap_reference / norm
        
        self.logger.debug(f"Updated global SHAP reference (round {round_number})")
        return self._global_shap_reference
    
    def _integrate_shap_with_trust(self, original_trust_scores: Dict[str, float],
                                  shap_alignment_scores: Dict[str, float],
                                  trust_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Integrate SHAP alignment scores with original trust scores.
        
        Implements: enhanced_trust = γ_shap * shap_alignment + (1 - γ_shap) * original_trust
        
        Args:
            original_trust_scores: Original trust scores from trust evaluation
            shap_alignment_scores: SHAP alignment scores [0, 1]
            trust_config: Trust configuration containing gamma_shap parameter
            
        Returns:
            Enhanced trust scores integrating SHAP alignment
        """
        # Get SHAP integration weight from configuration
        gamma_shap = trust_config.get('gamma_shap', 0.25)  # Default 25% weight for SHAP
        
        enhanced_scores = {}
        
        for client_id, original_trust in original_trust_scores.items():
            if client_id in shap_alignment_scores:
                # Combine SHAP alignment with original trust
                shap_score = shap_alignment_scores[client_id]
                enhanced_trust = (gamma_shap * shap_score + 
                                (1 - gamma_shap) * original_trust)
                enhanced_scores[client_id] = max(0.0, min(1.0, enhanced_trust))
                
                self.logger.debug(f"Client {client_id} trust enhancement: "
                                f"{original_trust:.3f} -> {enhanced_scores[client_id]:.3f} "
                                f"(SHAP: {shap_score:.3f})")
            else:
                # No SHAP data, use original trust score
                enhanced_scores[client_id] = original_trust
        
        return enhanced_scores
    
    def _update_global_reference_fingerprint(self, metrics_list: List[Dict[str, Any]], 
                                           trusted_clients: List[str]) -> None:
        """
        Update the global reference fingerprint using only trusted clients.
        
        This ensures the global reference reflects the fingerprint patterns of 
        trusted clients, improving alignment detection for future rounds.
        
        Args:
            metrics_list: List of metrics dictionaries containing SHAP fingerprints
            trusted_clients: List of client IDs that survived trust filtering
        """
        try:
            # Extract fingerprints from trusted clients only
            trusted_fingerprints = {}
            
            for metrics in metrics_list:
                if (isinstance(metrics, dict) and 
                    "client_id" in metrics and 
                    "shap" in metrics and
                    metrics["client_id"] in trusted_clients):
                    
                    client_id = metrics["client_id"]
                    shap_fingerprint = metrics["shap"]
                    
                    if isinstance(shap_fingerprint, list) and len(shap_fingerprint) > 0:
                        fingerprint_array = np.array(shap_fingerprint, dtype=np.float32)
                        
                        # Validate and normalize
                        if not (np.any(np.isnan(fingerprint_array)) or np.any(np.isinf(fingerprint_array))):
                            norm = np.linalg.norm(fingerprint_array)
                            if norm > 1e-8:
                                trusted_fingerprints[client_id] = fingerprint_array / norm
            
            if trusted_fingerprints and hasattr(self, '_global_shap_reference'):
                # Update global reference using trusted clients only
                trusted_avg = np.mean(list(trusted_fingerprints.values()), axis=0)
                
                # Blend with existing reference (higher weight on trusted clients)
                alpha_trusted = 0.3  # Higher learning rate for trusted updates
                self._global_shap_reference = (alpha_trusted * trusted_avg + 
                                             (1 - alpha_trusted) * self._global_shap_reference)
                
                # Re-normalize
                norm = np.linalg.norm(self._global_shap_reference)
                if norm > 1e-8:
                    self._global_shap_reference = self._global_shap_reference / norm
                
                self.logger.debug(f"Updated global SHAP reference using {len(trusted_fingerprints)} trusted clients")
            
        except Exception as e:
            self.logger.warning(f"Failed to update global reference fingerprint: {e}")
    
    def get_shap_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive SHAP alignment statistics for monitoring.
        
        Returns:
            Dictionary containing SHAP-related statistics and insights
        """
        stats = {
            'shap_enabled': hasattr(self, '_global_shap_reference') and self._global_shap_reference is not None,
            'global_reference_dimensions': None,
            'global_reference_norm': None
        }
        
        if hasattr(self, '_global_shap_reference') and self._global_shap_reference is not None:
            stats['global_reference_dimensions'] = len(self._global_shap_reference)
            stats['global_reference_norm'] = float(np.linalg.norm(self._global_shap_reference))
            
            # Additional insights about the global reference
            stats['global_reference_stats'] = {
                'mean': float(np.mean(self._global_shap_reference)),
                'std': float(np.std(self._global_shap_reference)),
                'min': float(np.min(self._global_shap_reference)),
                'max': float(np.max(self._global_shap_reference)),
                'sparsity': float(np.sum(np.abs(self._global_shap_reference) < 1e-6) / len(self._global_shap_reference))
            }
        
        return stats
    
    # ========================= DYNAMIC THRESHOLD MECHANISM =========================
    
    def __init_dynamic_threshold_system(self):
        """Initialize dynamic threshold system components."""
        if not hasattr(self, '_dynamic_threshold_initialized'):
            # Dynamic threshold configuration
            self.min_threshold = self.config.get('min_trust_threshold', 0.1)
            self.max_threshold = self.config.get('max_trust_threshold', 0.9)
            self.min_trusted_clients = self.config.get('min_trusted_clients', 2)
            self.target_trusted_ratio = self.config.get('target_trusted_ratio', 0.6)
            
            # History tracking for dynamic adaptation
            self.threshold_history = []
            self.trust_scores_history = []
            self.performance_history = []
            self.round_number_history = []
            
            # Threshold calculation weights
            self.percentile_weight = self.config.get('threshold_percentile_weight', 0.4)
            self.statistical_weight = self.config.get('threshold_statistical_weight', 0.3)
            self.adaptive_weight = self.config.get('threshold_adaptive_weight', 0.3)
            
            self._dynamic_threshold_initialized = True
            self.logger.info("Dynamic threshold system initialized")
    
    def calculate_dynamic_threshold(self, trust_scores: Dict[str, float], round_number: int, 
                                   global_accuracy: Optional[float] = None) -> float:
        """
        Calculate dynamic trust threshold based on current trust distribution and performance.
        
        Args:
            trust_scores: Current trust scores for all clients
            round_number: Current federated learning round
            global_accuracy: Current global model accuracy for performance tracking
            
        Returns:
            Dynamic trust threshold
        """
        # Initialize dynamic threshold system if not already done
        self.__init_dynamic_threshold_system()
        
        if not trust_scores:
            self.logger.warning("No trust scores provided - using minimum threshold")
            return self.min_threshold
        
        scores = list(trust_scores.values())
        num_clients = len(scores)
        
        self.logger.info(f"Round {round_number}: Calculating dynamic threshold for {num_clients} clients")
        
        # Method 1: Percentile-based threshold
        threshold_percentile = self._calculate_percentile_threshold(scores, round_number)
        
        # Method 2: Statistical-based threshold  
        threshold_statistical = self._calculate_statistical_threshold(scores, round_number)
        
        # Method 3: Adaptive threshold based on history
        threshold_adaptive = self._calculate_adaptive_threshold(scores, round_number)
        
        # Combine methods with round-aware weights
        if round_number <= 3:
            # Early rounds: be more lenient, focus on percentile and statistical
            dynamic_threshold = (
                0.5 * threshold_percentile + 
                0.4 * threshold_statistical + 
                0.1 * threshold_adaptive
            )
            self.logger.debug(f"Round {round_number}: Using early-round threshold calculation")
        elif round_number <= 10:
            # Middle rounds: balanced approach
            dynamic_threshold = (
                self.percentile_weight * threshold_percentile + 
                self.statistical_weight * threshold_statistical + 
                self.adaptive_weight * threshold_adaptive
            )
            self.logger.debug(f"Round {round_number}: Using balanced threshold calculation")
        else:
            # Later rounds: more sophisticated, history-based
            dynamic_threshold = (
                0.2 * threshold_percentile + 
                0.3 * threshold_statistical + 
                0.5 * threshold_adaptive
            )
            self.logger.debug(f"Round {round_number}: Using advanced threshold calculation")
        
        # Apply hard constraints
        dynamic_threshold = max(self.min_threshold, min(dynamic_threshold, self.max_threshold))
        
        # Ensure minimum number of trusted clients
        dynamic_threshold = self._ensure_minimum_trusted_clients(scores, dynamic_threshold)
        
        # Store for history
        self.threshold_history.append(dynamic_threshold)
        self.trust_scores_history.append(scores.copy())
        self.round_number_history.append(round_number)
        if global_accuracy is not None:
            self.performance_history.append(global_accuracy)
        
        # Log detailed threshold calculation
        trusted_count = sum(1 for score in scores if score >= dynamic_threshold)
        trusted_ratio = trusted_count / num_clients
        
        self.logger.info(f"Round {round_number} Dynamic Threshold Calculation:")
        self.logger.info(f"  - Percentile threshold: {threshold_percentile:.3f}")
        self.logger.info(f"  - Statistical threshold: {threshold_statistical:.3f}")
        self.logger.info(f"  - Adaptive threshold: {threshold_adaptive:.3f}")
        self.logger.info(f"  - Final dynamic threshold: {dynamic_threshold:.3f}")
        self.logger.info(f"  - Trusted clients: {trusted_count}/{num_clients} ({trusted_ratio:.1%})")
        self.logger.info(f"  - Trust score range: [{min(scores):.3f}, {max(scores):.3f}]")
        
        # Update the threshold for current operations
        self.threshold = dynamic_threshold
        
        return dynamic_threshold
    
    def _calculate_percentile_threshold(self, scores: List[float], round_number: int) -> float:
        """Calculate threshold based on score percentiles."""
        scores_sorted = sorted(scores, reverse=True)
        
        # Adaptive percentile based on round number
        if round_number <= 2:
            percentile = 0.8  # Top 80% in early rounds
        elif round_number <= 5:
            percentile = 0.7  # Top 70% in middle rounds
        elif round_number <= 10:
            percentile = self.target_trusted_ratio  # Target ratio
        else:
            # Later rounds: more selective
            percentile = max(0.5, self.target_trusted_ratio - 0.1)
        
        index = int(len(scores_sorted) * percentile)
        index = max(0, min(index, len(scores_sorted) - 1))
        
        return scores_sorted[index]
    
    def _calculate_statistical_threshold(self, scores: List[float], round_number: int) -> float:
        """Calculate threshold based on statistical properties."""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Adaptive statistical threshold
        if round_number <= 3:
            # Early rounds: mean - 0.5 * std (more inclusive)
            statistical_threshold = mean_score - 0.5 * std_score
        elif round_number <= 10:
            # Middle rounds: mean (50th percentile)
            statistical_threshold = mean_score
        else:
            # Later rounds: mean + 0.2 * std (more selective)
            statistical_threshold = mean_score + 0.2 * std_score
        
        return max(0.0, statistical_threshold)
    
    def _calculate_adaptive_threshold(self, scores: List[float], round_number: int) -> float:
        """Calculate adaptive threshold based on historical performance."""
        if len(self.threshold_history) < 2:
            return np.mean(scores)
        
        # Analyze threshold effectiveness over recent rounds
        recent_window = min(3, len(self.threshold_history))
        recent_thresholds = self.threshold_history[-recent_window:]
        recent_scores = self.trust_scores_history[-recent_window:]
        
        # Calculate trusted ratios for recent rounds
        trusted_ratios = []
        for threshold, round_scores in zip(recent_thresholds, recent_scores):
            trusted_count = sum(1 for score in round_scores if score >= threshold)
            trusted_ratios.append(trusted_count / len(round_scores))
        
        avg_trusted_ratio = np.mean(trusted_ratios)
        current_mean = np.mean(scores)
        current_std = np.std(scores)
        
        # Analyze performance trend if available
        performance_trend = 0.0
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            performance_trend = recent_performance[-1] - recent_performance[0]
        
        # Adjust based on recent performance and trusted ratio
        if avg_trusted_ratio > 0.8:
            # Too inclusive - increase threshold
            adaptive_threshold = current_mean + 0.3 * current_std
            self.logger.debug("Adaptive: Too many trusted clients - increasing threshold")
        elif avg_trusted_ratio < 0.3:
            # Too exclusive - decrease threshold
            adaptive_threshold = current_mean - 0.4 * current_std
            self.logger.debug("Adaptive: Too few trusted clients - decreasing threshold")
        elif performance_trend > 0.05:
            # Performance improving - can be more selective
            adaptive_threshold = current_mean + 0.1 * current_std
            self.logger.debug("Adaptive: Performance improving - slightly increasing threshold")
        elif performance_trend < -0.05:
            # Performance degrading - be more inclusive
            adaptive_threshold = current_mean - 0.2 * current_std
            self.logger.debug("Adaptive: Performance degrading - decreasing threshold")
        else:
            # Stable performance - maintain current level
            adaptive_threshold = current_mean
            self.logger.debug("Adaptive: Stable performance - maintaining current level")
        
        return max(0.0, adaptive_threshold)
    
    def _ensure_minimum_trusted_clients(self, scores: List[float], threshold: float) -> float:
        """Ensure at least minimum number of clients will be trusted."""
        scores_sorted = sorted(scores, reverse=True)
        trusted_count = sum(1 for score in scores if score >= threshold)
        
        if trusted_count < self.min_trusted_clients:
            if len(scores_sorted) >= self.min_trusted_clients:
                # Adjust threshold to ensure minimum trusted clients
                adjusted_threshold = scores_sorted[self.min_trusted_clients - 1]
                self.logger.info(f"Adjusted threshold from {threshold:.3f} to {adjusted_threshold:.3f} "
                               f"to ensure {self.min_trusted_clients} trusted clients")
                return adjusted_threshold
            else:
                # Not enough clients total
                self.logger.warning(f"Only {len(scores)} clients total, cannot ensure {self.min_trusted_clients} trusted clients")
                return self.min_threshold
        
        return threshold
    
    def get_trusted_clients_dynamic(self, trust_scores: Dict[str, float], round_number: int,
                                   global_accuracy: Optional[float] = None) -> Tuple[Dict[str, float], float]:
        """
        Get trusted clients using dynamic threshold calculation.
        
        Args:
            trust_scores: Trust scores for all clients
            round_number: Current federated learning round
            global_accuracy: Current global model accuracy
            
        Returns:
            Tuple of (trusted_clients_dict, dynamic_threshold_used)
        """
        # Calculate dynamic threshold
        dynamic_threshold = self.calculate_dynamic_threshold(trust_scores, round_number, global_accuracy)
        
        # Filter trusted clients
        trusted_clients = {
            client_id: score for client_id, score in trust_scores.items() 
            if score >= dynamic_threshold
        }
        
        self.logger.info(f"Round {round_number}: {len(trusted_clients)}/{len(trust_scores)} clients trusted "
                        f"(dynamic threshold: {dynamic_threshold:.3f})")
        
        # Log trust distribution for analysis
        scores = list(trust_scores.values())
        self.logger.info(f"Trust scores - Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}, "
                        f"Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
        
        # Fallback if no clients are trusted (shouldn't happen with dynamic threshold, but safety)
        if not trusted_clients:
            self.logger.warning("Dynamic threshold resulted in no trusted clients - using fallback")
            # Use top 50% of clients as fallback
            sorted_clients = sorted(trust_scores.items(), key=lambda x: x[1], reverse=True)
            fallback_count = max(1, len(sorted_clients) // 2)
            trusted_clients = dict(sorted_clients[:fallback_count])
            self.logger.info(f"Fallback: Selected top {len(trusted_clients)} clients")
        
        return trusted_clients, dynamic_threshold
    
    def update_performance_history(self, global_accuracy: float, round_number: int):
        """Update performance history for adaptive threshold calculation."""
        self.performance_history.append(global_accuracy)
        
        # Keep only recent history
        max_history = 15
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            
        self.logger.debug(f"Round {round_number}: Updated performance history with accuracy {global_accuracy:.3f}")
    
    def get_threshold_statistics(self) -> Dict[str, Any]:
        """Get statistics about dynamic threshold behavior."""
        if not hasattr(self, '_dynamic_threshold_initialized') or not self.threshold_history:
            return {'dynamic_threshold_enabled': False}
        
        return {
            'dynamic_threshold_enabled': True,
            'current_threshold': self.threshold,
            'threshold_history': self.threshold_history[-10:],  # Last 10 rounds
            'threshold_stats': {
                'mean': np.mean(self.threshold_history),
                'std': np.std(self.threshold_history),
                'min': np.min(self.threshold_history),
                'max': np.max(self.threshold_history),
                'trend': self.threshold_history[-1] - self.threshold_history[0] if len(self.threshold_history) > 1 else 0
            },
            'configuration': {
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'min_trusted_clients': self.min_trusted_clients,
                'target_trusted_ratio': self.target_trusted_ratio
            }
        }
