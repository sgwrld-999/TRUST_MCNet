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
from typing import Dict, List, Any, Union
from scipy.stats import entropy
from collections import defaultdict
import logging


class TrustEvaluator:
    """
    Comprehensive trust evaluation system for federated learning clients.
    
    Supports multiple trust evaluation modes:
    - 'cosine': Cosine similarity between model updates
    - 'entropy': Entropy-based trust evaluation
    - 'reputation': Historical performance-based reputation
    - 'hybrid': Combination of multiple trust metrics
    """
    
    def __init__(self, trust_mode: str = 'hybrid', threshold: float = 0.5):
        """
        Initialize trust evaluator.
        
        Args:
            trust_mode: Trust evaluation method ('cosine', 'entropy', 'reputation', 'hybrid')
            threshold: Trust threshold for client selection
        """
        self.trust_mode = trust_mode
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Historical data for reputation calculation
        self.client_history = defaultdict(list)
        self.global_update_history = []
        
        # Hyperparameters for hybrid trust
        self.weights = {
            'cosine': 0.4,
            'entropy': 0.3,
            'reputation': 0.3
        }
    
    def evaluate_trust(self, client_id: str, model_update: Dict[str, torch.Tensor],
                      performance_metrics: Dict[str, float], 
                      global_model: Dict[str, torch.Tensor],
                      round_number: int) -> float:
        """
        Evaluate trust score for a client based on their model update.
        
        Args:
            client_id: Unique identifier for the client
            model_update: Client's model parameter updates
            performance_metrics: Client's performance metrics (accuracy, loss, etc.)
            global_model: Current global model parameters
            round_number: Current federated learning round
            
        Returns:
            Trust score between 0 and 1
        """
        if self.trust_mode == 'cosine':
            return self._cosine_trust(model_update, global_model)
        elif self.trust_mode == 'entropy':
            return self._entropy_trust(model_update)
        elif self.trust_mode == 'reputation':
            return self._reputation_trust(client_id, performance_metrics, round_number)
        elif self.trust_mode == 'hybrid':
            return self._hybrid_trust(client_id, model_update, performance_metrics, 
                                    global_model, round_number)
        else:
            raise ValueError(f"Unknown trust mode: {self.trust_mode}")
    
    def _cosine_trust(self, model_update: Dict[str, torch.Tensor], 
                     global_model: Dict[str, torch.Tensor]) -> float:
        """
        Calculate trust based on cosine similarity between client update and global model.
        
        Args:
            model_update: Client's model parameters
            global_model: Global model parameters
            
        Returns:
            Cosine similarity-based trust score
        """
        # Flatten model parameters
        client_params = torch.cat([param.flatten() for param in model_update.values()])
        global_params = torch.cat([param.flatten() for param in global_model.values()])
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(client_params.unsqueeze(0), 
                                       global_params.unsqueeze(0))
        
        # Convert to trust score (0 to 1 range)
        trust_score = (cosine_sim.item() + 1) / 2
        
        return max(0.0, min(1.0, trust_score))
    
    def _entropy_trust(self, model_update: Dict[str, torch.Tensor]) -> float:
        """
        Calculate trust based on entropy of model parameter distributions.
        
        Args:
            model_update: Client's model parameters
            
        Returns:
            Entropy-based trust score
        """
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
        # Assume max entropy around 4.0 for normalization
        trust_score = min(1.0, avg_entropy / 4.0)
        
        return max(0.0, trust_score)
    
    def _reputation_trust(self, client_id: str, performance_metrics: Dict[str, float],
                         round_number: int) -> float:
        """
        Calculate trust based on historical performance (reputation).
        
        Args:
            client_id: Client identifier
            performance_metrics: Current round performance metrics
            round_number: Current round number
            
        Returns:
            Reputation-based trust score
        """
        # Store current performance
        current_performance = performance_metrics.get('accuracy', 0.0)
        self.client_history[client_id].append({
            'round': round_number,
            'accuracy': current_performance,
            'loss': performance_metrics.get('loss', 1.0),
            'f1_score': performance_metrics.get('f1_score', 0.0)
        })
        
        # Calculate reputation based on historical performance
        history = self.client_history[client_id]
        
        if len(history) < 2:
            # Not enough history, use current performance
            return current_performance
        
        # Calculate weighted average with recent performance having higher weight
        weights = np.exp(np.linspace(-1, 0, len(history)))  # Exponential decay
        weights = weights / weights.sum()
        
        accuracies = [entry['accuracy'] for entry in history]
        reputation_score = np.average(accuracies, weights=weights)
        
        # Consider consistency (lower variance = higher trust)
        consistency = 1.0 / (1.0 + np.var(accuracies))
        
        # Combine reputation and consistency
        trust_score = 0.7 * reputation_score + 0.3 * consistency
        
        return max(0.0, min(1.0, trust_score))
    
    def _hybrid_trust(self, client_id: str, model_update: Dict[str, torch.Tensor],
                     performance_metrics: Dict[str, float], 
                     global_model: Dict[str, torch.Tensor],
                     round_number: int) -> float:
        """
        Calculate hybrid trust combining multiple trust metrics.
        
        Args:
            client_id: Client identifier
            model_update: Client's model parameters
            performance_metrics: Current round performance metrics
            global_model: Global model parameters
            round_number: Current round number
            
        Returns:
            Hybrid trust score
        """
        # Calculate individual trust components
        cosine_trust = self._cosine_trust(model_update, global_model)
        entropy_trust = self._entropy_trust(model_update)
        reputation_trust = self._reputation_trust(client_id, performance_metrics, round_number)
        
        # Weighted combination
        hybrid_score = (self.weights['cosine'] * cosine_trust +
                       self.weights['entropy'] * entropy_trust +
                       self.weights['reputation'] * reputation_trust)
        
        # Log individual components for debugging
        self.logger.debug(f"Client {client_id} trust components - "
                         f"Cosine: {cosine_trust:.3f}, "
                         f"Entropy: {entropy_trust:.3f}, "
                         f"Reputation: {reputation_trust:.3f}, "
                         f"Hybrid: {hybrid_score:.3f}")
        
        return max(0.0, min(1.0, hybrid_score))
    
    def select_trusted_clients(self, available_clients: List[str],
                              client_trust_scores: Dict[str, List[float]],
                              selection_ratio: float = 0.8) -> List[str]:
        """
        Select trusted clients based on their trust scores.
        
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
        
        # Filter clients above threshold
        trusted_clients = [client_id for client_id, score in client_scores.items() 
                          if score >= self.threshold]
        
        if not trusted_clients:
            # If no clients meet threshold, select top performers
            trusted_clients = sorted(client_scores.keys(), 
                                   key=lambda x: client_scores[x], 
                                   reverse=True)[:max(1, int(len(available_clients) * 0.5))]
        
        # Select based on ratio
        num_selected = max(1, int(len(trusted_clients) * selection_ratio))
        
        # Probability-based selection weighted by trust scores
        if len(trusted_clients) <= num_selected:
            selected = trusted_clients
        else:
            # Weighted random selection
            scores = np.array([client_scores[client_id] for client_id in trusted_clients])
            probabilities = scores / scores.sum()
            
            selected = np.random.choice(
                trusted_clients,
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
