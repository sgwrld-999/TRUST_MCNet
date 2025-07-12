"""
Enhanced Federated Server with FLTrust integration for TRUST-MCNet.

This module extends the base FederatedServer to properly support FLTrust
aggregation with the required server update mechanism: w_{t+1} = w_t + η · Δw_global
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from collections import defaultdict

from ..server.server import FederatedServer
from ..utils.aggregator import FLTrustAggregator, AggregatorInterface
from ..trust_module.trust_evaluator import TrustEvaluator


class FLTrustFederatedServer(FederatedServer):
    """
    Enhanced Federated Server with proper FLTrust integration.
    
    Extends the base TRUST-MCNet server to support FLTrust aggregation
    with correct server-side update mechanism and learning rate application.
    """
    
    def __init__(self, global_model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the FLTrust-enabled federated server.
        
        Args:
            global_model: The initial global model architecture
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__(global_model, config)
        
        # FLTrust-specific configuration
        self.server_learning_rate = config.get('server_learning_rate', 1.0)
        self.use_fltrust_updates = False  # Will be set to True when FLTrust aggregator is used
        
        self.logger.info(f"FLTrust server initialized with learning rate: {self.server_learning_rate}")
    
    def set_aggregator(self, aggregator: AggregatorInterface):
        """
        Set the aggregation strategy for the server.
        
        Args:
            aggregator: Aggregator instance (FLTrustAggregator, FedAvgAggregator, etc.)
        """
        self.aggregator = aggregator
        
        # Check if using FLTrust aggregator
        self.use_fltrust_updates = isinstance(aggregator, FLTrustAggregator)
        
        if self.use_fltrust_updates:
            self.logger.info("Server configured for FLTrust aggregation with delta updates")
        else:
            self.logger.info(f"Server configured for {type(aggregator).__name__} aggregation")
    
    def aggregate_models(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                        client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates with proper FLTrust integration.
        
        Args:
            client_updates: Dictionary mapping client_id to model parameters
            client_metrics: Dictionary mapping client_id to performance metrics
            
        Returns:
            Aggregated model parameters or updates (depending on aggregator type)
        """
        # Calculate trust scores for participating clients (if using TRUST-MCNet evaluation)
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
        
        # Aggregate based on aggregator type
        if self.use_fltrust_updates:
            # FLTrust expects client weights and returns aggregated update (Δw_global)
            aggregated_result = self.aggregator.aggregate(
                client_updates=client_updates,
                weights=None,  # FLTrust computes its own trust scores
                global_weights=self.global_model.state_dict(),
                trust_scores_override=current_trust_scores  # Use TRUST-MCNet scores
            )
            return aggregated_result
        else:
            # Other aggregators expect updates and return aggregated parameters
            aggregated_result = self.aggregator.aggregate(
                client_updates,
                weights=current_trust_scores
            )
            return aggregated_result
    
    def update_global_model(self, aggregated_result: Dict[str, torch.Tensor]):
        """
        Update the global model with proper handling for FLTrust delta updates.
        
        Args:
            aggregated_result: Aggregated parameters or updates from aggregator
        """
        if self.use_fltrust_updates:
            # For FLTrust: w_{t+1} = w_t + η · Δw_global
            current_weights = self.global_model.state_dict()
            updated_weights = {}
            
            for param_name, param_value in current_weights.items():
                if param_name in aggregated_result:
                    # Apply server learning rate to the aggregated update
                    delta = aggregated_result[param_name]
                    updated_weights[param_name] = param_value + self.server_learning_rate * delta
                else:
                    # Keep parameter unchanged if not in aggregated result
                    updated_weights[param_name] = param_value
            
            self.global_model.load_state_dict(updated_weights)
            self.logger.info(f"Global model updated with FLTrust deltas (η={self.server_learning_rate}) for round {self.round_number}")
        else:
            # For other aggregators: direct parameter replacement
            self.global_model.load_state_dict(aggregated_result)
            self.logger.info(f"Global model updated with aggregated parameters for round {self.round_number}")
    
    def run_federated_round(self, available_clients: List[str], 
                           client_trainer_fn) -> Dict[str, Any]:
        """
        Execute one complete federated learning round with FLTrust support.
        
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
        
        # Step 4: Aggregate models (handles both FLTrust and other aggregators)
        if client_updates:
            aggregated_result = self.aggregate_models(client_updates, client_metrics)
            
            # Step 5: Update global model (handles FLTrust delta updates correctly)
            self.update_global_model(aggregated_result)
        
        # Step 6: Prepare round results
        round_results = {
            'round': self.round_number,
            'selected_clients': selected_clients,
            'participating_clients': list(client_updates.keys()),
            'trust_scores': {k: v[-1] for k, v in self.client_trust_scores.items() if v},
            'avg_client_metrics': self._average_client_metrics(client_metrics),
            'aggregator_type': type(self.aggregator).__name__,
            'server_learning_rate': self.server_learning_rate if self.use_fltrust_updates else None
        }
        
        self.round_number += 1
        return round_results
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server configuration information."""
        base_info = {
            'round_number': self.round_number,
            'aggregator_type': type(self.aggregator).__name__,
            'trust_mode': self.trust_evaluator.trust_mode,
            'trust_threshold': self.trust_evaluator.threshold,
            'server_learning_rate': self.server_learning_rate,
            'use_fltrust_updates': self.use_fltrust_updates
        }
        
        # Add aggregator-specific info if available
        if hasattr(self.aggregator, 'get_aggregation_info'):
            base_info['aggregator_info'] = self.aggregator.get_aggregation_info()
        
        return base_info


def create_fltrust_server(global_model: nn.Module, 
                         root_dataset_loader: torch.utils.data.DataLoader,
                         config: Dict[str, Any]) -> FLTrustFederatedServer:
    """
    Convenience function to create a FLTrust-enabled server.
    
    Args:
        global_model: Global model architecture
        root_dataset_loader: Trusted root dataset for FLTrust
        config: Server configuration
        
    Returns:
        Configured FLTrust server
    """
    # Create server
    server = FLTrustFederatedServer(global_model, config)
    
    # Create root model (copy of global model)
    root_model = type(global_model)(**config.get('model_params', {}))
    root_model.load_state_dict(global_model.state_dict())
    
    # Create FLTrust aggregator
    fltrust_aggregator = FLTrustAggregator(
        root_model=root_model,
        root_dataset_loader=root_dataset_loader,
        server_learning_rate=config.get('server_learning_rate', 1.0),
        clip_threshold=config.get('clip_threshold', 10.0),
        warm_up_rounds=config.get('warm_up_rounds', 5),
        trust_threshold=config.get('fltrust_trust_threshold', 0.0)
    )
    
    # Set aggregator
    server.set_aggregator(fltrust_aggregator)
    
    logging.getLogger(__name__).info("FLTrust server created and configured successfully")
    
    return server
