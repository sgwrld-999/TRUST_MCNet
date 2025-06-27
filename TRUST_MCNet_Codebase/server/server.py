"""
Central server for TRUST-MCNet federated learning framework.

This module implements the central controller that handles:
- Trust evaluation of clients
- Model aggregation with trust-based weighting
- Broadcasting global model updates
- Coordinating federated training rounds
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from collections import defaultdict

from ..trust_module.trust_evaluator import TrustEvaluator
from ..utils.aggregator import TrimmedMeanAggregator
from ..utils.metrics import ModelMetrics


class FederatedServer:
    """
    Central server for TRUST-MCNet federated learning.
    
    Manages client trust evaluation, model aggregation, and coordination
    of federated training rounds with trust-based client selection.
    """
    
    def __init__(self, global_model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the federated server.
        
        Args:
            global_model: The initial global model architecture
            config: Configuration dictionary containing hyperparameters
        """
        self.global_model = global_model
        self.config = config
        self.round_number = 0
        
        # Initialize trust evaluator
        self.trust_evaluator = TrustEvaluator(
            trust_mode=config.get('trust_mode', 'hybrid'),
            threshold=config.get('trust_threshold', 0.5)
        )
        
        # Initialize aggregator
        self.aggregator = TrimmedMeanAggregator(
            trim_ratio=config.get('trim_ratio', 0.1)
        )
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Client tracking
        self.client_trust_scores = defaultdict(list)
        self.client_participation_history = defaultdict(list)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def select_clients(self, available_clients: List[str], 
                      selection_ratio: float = 0.8) -> List[str]:
        """
        Select clients for current training round based on trust scores.
        
        Args:
            available_clients: List of available client IDs
            selection_ratio: Fraction of clients to select
            
        Returns:
            List of selected client IDs
        """
        if self.round_number == 0:
            # First round: select randomly
            num_selected = max(1, int(len(available_clients) * selection_ratio))
            selected = np.random.choice(
                available_clients, 
                size=num_selected, 
                replace=False
            ).tolist()
        else:
            # Select based on trust scores
            selected = self.trust_evaluator.select_trusted_clients(
                available_clients,
                self.client_trust_scores,
                selection_ratio
            )
        
        self.logger.info(f"Round {self.round_number}: Selected {len(selected)} clients")
        return selected
    
    def aggregate_models(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                        client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using trust-weighted averaging.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            client_metrics: Dictionary mapping client_id to performance metrics
            
        Returns:
            Aggregated global model parameters
        """
        # Calculate trust scores for participating clients
        current_trust_scores = {}
        for client_id in client_updates.keys():
            trust_score = self.trust_evaluator.evaluate_trust(
                client_id=client_id,
                model_update=client_updates[client_id],
                performance_metrics=client_metrics[client_id],
                global_model=self.global_model.state_dict(),
                round_number=self.round_number
            )
            current_trust_scores[client_id] = trust_score
            self.client_trust_scores[client_id].append(trust_score)
        
        # Log trust scores
        trust_summary = {k: f"{v:.3f}" for k, v in current_trust_scores.items()}
        self.logger.info(f"Trust scores for round {self.round_number}: {trust_summary}")
        
        # Aggregate using trust-weighted approach
        aggregated_params = self.aggregator.aggregate(
            client_updates,
            weights=current_trust_scores
        )
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """
        Update the global model with aggregated parameters.
        
        Args:
            aggregated_params: Aggregated model parameters
        """
        self.global_model.load_state_dict(aggregated_params)
        self.logger.info(f"Global model updated for round {self.round_number}")
    
    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """
        Broadcast current global model parameters to clients.
        
        Returns:
            Current global model state dictionary
        """
        return self.global_model.state_dict().copy()
    
    def run_federated_round(self, available_clients: List[str], 
                           client_trainer_fn) -> Dict[str, Any]:
        """
        Execute one complete federated learning round.
        
        Args:
            available_clients: List of available client IDs
            client_trainer_fn: Function to train clients
            
        Returns:
            Round results including metrics and trust scores
        """
        self.logger.info(f"Starting federated round {self.round_number}")
        
        # Step 1: Select clients
        selected_clients = self.select_clients(available_clients)
        
        # Step 2: Broadcast global model
        global_params = self.broadcast_model()
        
        # Step 3: Train selected clients
        client_updates = {}
        client_metrics = {}
        
        for client_id in selected_clients:
            try:
                update, metrics = client_trainer_fn(client_id, global_params)
                client_updates[client_id] = update
                client_metrics[client_id] = metrics
                self.client_participation_history[client_id].append(self.round_number)
            except Exception as e:
                self.logger.warning(f"Client {client_id} failed training: {e}")
        
        # Step 4: Aggregate models
        if client_updates:
            aggregated_params = self.aggregate_models(client_updates, client_metrics)
            
            # Step 5: Update global model
            self.update_global_model(aggregated_params)
        
        # Step 6: Prepare round results
        round_results = {
            'round': self.round_number,
            'selected_clients': selected_clients,
            'participating_clients': list(client_updates.keys()),
            'trust_scores': {k: v[-1] for k, v in self.client_trust_scores.items() if v},
            'avg_client_metrics': self._average_client_metrics(client_metrics)
        }
        
        self.round_number += 1
        return round_results
    
    def _average_client_metrics(self, client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics across participating clients."""
        if not client_metrics:
            return {}
        
        avg_metrics = defaultdict(list)
        for metrics in client_metrics.values():
            for key, value in metrics.items():
                avg_metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in avg_metrics.items()}
    
    def get_trust_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive trust summary for all clients.
        
        Returns:
            Dictionary with trust statistics for each client
        """
        trust_summary = {}
        for client_id, scores in self.client_trust_scores.items():
            if scores:
                trust_summary[client_id] = {
                    'current_trust': scores[-1],
                    'avg_trust': np.mean(scores),
                    'trust_trend': scores[-1] - scores[0] if len(scores) > 1 else 0.0,
                    'participation_rounds': len(self.client_participation_history[client_id])
                }
        return trust_summary
    
    def save_checkpoint(self, filepath: str):
        """Save server state for resuming training."""
        checkpoint = {
            'global_model': self.global_model.state_dict(),
            'round_number': self.round_number,
            'client_trust_scores': dict(self.client_trust_scores),
            'client_participation_history': dict(self.client_participation_history)
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Server checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load server state from checkpoint."""
        checkpoint = torch.load(filepath)
        self.global_model.load_state_dict(checkpoint['global_model'])
        self.round_number = checkpoint['round_number']
        self.client_trust_scores = defaultdict(list, checkpoint['client_trust_scores'])
        self.client_participation_history = defaultdict(list, checkpoint['client_participation_history'])
        self.logger.info(f"Server checkpoint loaded from {filepath}")
