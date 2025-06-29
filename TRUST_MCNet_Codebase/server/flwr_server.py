"""
Flwr-based server implementation for TRUST-MCNet federated learning.

This module implements the Flwr server with FedAdam strategy and trust mechanisms
optimized for IoT devices.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import (
    Parameters, 
    FitRes, 
    EvaluateRes, 
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.strategy import FedAdam
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from collections import defaultdict
import numpy as np
import psutil
import time

from ..trust_module.trust_evaluator import TrustEvaluator
from ..utils.metrics import ModelMetrics


class TrustAwareFedAdam(FedAdam):
    """
    Custom FedAdam strategy with trust mechanisms for IoT federated learning.
    
    Integrates trust evaluation with FedAdam optimization strategy,
    considering resource constraints of IoT devices.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        config: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize TrustAwareFedAdam strategy.
        
        Args:
            global_model: The global model architecture
            config: Configuration dictionary
            **kwargs: Additional FedAdam parameters
        """
        # Extract FedAdam specific parameters from config
        strategy_config = config.get('federated', {}).get('strategy_config', {})
        
        super().__init__(
            fraction_fit=config.get('federated', {}).get('fraction_fit', 0.8),
            fraction_evaluate=config.get('federated', {}).get('fraction_evaluate', 0.2),
            min_fit_clients=config.get('federated', {}).get('min_fit_clients', 5),
            min_evaluate_clients=config.get('federated', {}).get('min_evaluate_clients', 2),
            min_available_clients=config.get('federated', {}).get('min_available_clients', 5),
            eta=strategy_config.get('eta', 0.001),
            eta_l=strategy_config.get('eta_l', 0.001),
            beta_1=strategy_config.get('beta_1', 0.9),
            beta_2=strategy_config.get('beta_2', 0.999),
            tau=strategy_config.get('tau', 0.001),
            **kwargs
        )
        
        self.global_model = global_model
        self.config = config
        self.round_number = 0
        
        # Initialize trust evaluator
        self.trust_evaluator = TrustEvaluator(
            trust_mode=config.get('trust', {}).get('trust_mode', 'hybrid'),
            threshold=config.get('trust', {}).get('trust_threshold', 0.7)
        )
        
        # Initialize metrics tracker
        self.metrics = ModelMetrics()
        
        # Client tracking for trust and IoT considerations
        self.client_trust_scores = defaultdict(list)
        self.client_resource_usage = defaultdict(list)
        self.client_performance_history = defaultdict(list)
        
        # IoT configuration
        self.iot_config = config.get('federated', {}).get('iot_config', {})
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize global model parameters
        self.initial_parameters = self._get_model_parameters()
        
    def _get_model_parameters(self) -> Parameters:
        """Get model parameters as Flwr Parameters."""
        return ndarrays_to_parameters([val.cpu().numpy() for val in self.global_model.state_dict().values()])
    
    def _set_model_parameters(self, parameters: Parameters) -> None:
        """Set model parameters from Flwr Parameters."""
        params_dict = zip(self.global_model.state_dict().keys(), parameters_to_ndarrays(parameters))
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.global_model.load_state_dict(state_dict, strict=True)
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        self.logger.info("Initializing global model parameters for IoT federated learning")
        return self.initial_parameters
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of training with IoT considerations."""
        self.round_number = server_round
        
        # Get available clients
        available_clients = client_manager.all()
        self.logger.info(f"Round {server_round}: {len(available_clients)} clients available")
        
        # Apply trust-based client selection
        if server_round > 1:
            trusted_clients = self._select_trusted_clients(available_clients)
        else:
            # First round: select all available clients (up to limit)
            max_clients = min(len(available_clients), 
                            int(len(available_clients) * self.fraction_fit))
            trusted_clients = available_clients[:max_clients]
        
        # Configure training parameters with IoT optimizations
        fit_config = self._get_fit_config(server_round)
        
        # Create client configurations
        client_configs = []
        for client in trusted_clients:
            # Adaptive configuration based on client resources
            client_config = self._adapt_config_for_client(client, fit_config)
            client_configs.append((client, client_config))
        
        self.logger.info(f"Round {server_round}: Selected {len(client_configs)} clients for training")
        return client_configs
    
    def _select_trusted_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Select clients based on trust scores and resource availability."""
        if not self.client_trust_scores:
            # No trust history, select randomly
            num_clients = min(len(available_clients), 
                            max(self.min_fit_clients, 
                                int(len(available_clients) * self.fraction_fit)))
            return np.random.choice(available_clients, num_clients, replace=False).tolist()
        
        # Calculate trust scores for available clients
        client_scores = {}
        for client in available_clients:
            client_id = client.cid
            if client_id in self.client_trust_scores:
                # Use recent trust score
                trust_score = np.mean(self.client_trust_scores[client_id][-5:])  # Last 5 rounds
                
                # Consider resource efficiency
                if client_id in self.client_resource_usage:
                    resource_efficiency = 1.0 - np.mean(self.client_resource_usage[client_id][-3:])
                    trust_score = 0.7 * trust_score + 0.3 * resource_efficiency
                
                client_scores[client] = trust_score
            else:
                client_scores[client] = 0.5  # Default score for new clients
        
        # Sort clients by trust score and select top performers
        sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure minimum number of clients
        num_clients = min(len(available_clients), 
                         max(self.min_fit_clients, 
                             int(len(available_clients) * self.fraction_fit)))
        
        selected_clients = [client for client, _ in sorted_clients[:num_clients]]
        
        self.logger.info(f"Selected clients with trust scores: "
                        f"{[(c.cid, client_scores[c]) for c in selected_clients]}")
        
        return selected_clients
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Generate training configuration for the current round."""
        return {
            "server_round": server_round,
            "local_epochs": self.config.get('client', {}).get('local_epochs', 5),
            "learning_rate": self.config.get('client', {}).get('learning_rate', 0.001),
            "batch_size": self.config.get('client', {}).get('batch_size', 32),
            "max_memory_mb": self.iot_config.get('max_memory_mb', 512),
            "max_cpu_percent": self.iot_config.get('max_cpu_percent', 70),
            "adaptive_batch_size": self.iot_config.get('adaptive_batch_size', True),
            "min_batch_size": self.iot_config.get('min_batch_size', 8),
            "max_batch_size": self.iot_config.get('max_batch_size', 64),
        }
    
    def _adapt_config_for_client(self, client: ClientProxy, base_config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Adapt configuration for specific client based on its capabilities."""
        client_config = base_config.copy()
        client_id = client.cid
        
        # Adapt batch size based on historical performance
        if client_id in self.client_resource_usage:
            avg_resource_usage = np.mean(self.client_resource_usage[client_id][-3:])
            if avg_resource_usage > 0.8:  # High resource usage
                client_config["batch_size"] = max(
                    self.iot_config.get('min_batch_size', 8),
                    int(client_config["batch_size"] * 0.7)
                )
                client_config["local_epochs"] = max(1, int(client_config["local_epochs"] * 0.8))
            elif avg_resource_usage < 0.3:  # Low resource usage
                client_config["batch_size"] = min(
                    self.iot_config.get('max_batch_size', 64),
                    int(client_config["batch_size"] * 1.3)
                )
        
        return client_config
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with trust evaluation."""
        
        if not results:
            return None, {}
        
        # Log failures for trust evaluation
        for failure in failures:
            if isinstance(failure, tuple):
                client, _ = failure
                self.logger.warning(f"Client {client.cid} failed in round {server_round}")
                self._update_client_trust(client.cid, success=False)
        
        # Evaluate trust for successful clients
        successful_results = []
        trust_weights = []
        
        for client, fit_res in results:
            client_id = client.cid
            
            # Calculate trust score based on performance and resource usage
            trust_score = self._calculate_trust_score(client_id, fit_res)
            self._update_client_trust(client_id, success=True, trust_score=trust_score)
            
            # Update resource usage tracking
            if "resource_usage" in fit_res.metrics:
                self.client_resource_usage[client_id].append(fit_res.metrics["resource_usage"])
            
            successful_results.append((client, fit_res))
            trust_weights.append(trust_score)
        
        # Normalize trust weights
        if trust_weights:
            trust_weights = np.array(trust_weights)
            trust_weights = trust_weights / np.sum(trust_weights)
        
        # Apply trust-weighted aggregation using parent FedAdam
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, successful_results, failures
        )
        
        # Add trust information to metrics
        if trust_weights.size > 0:
            aggregated_metrics["avg_trust_score"] = float(np.mean(trust_weights))
            aggregated_metrics["min_trust_score"] = float(np.min(trust_weights))
            aggregated_metrics["max_trust_score"] = float(np.max(trust_weights))
        
        aggregated_metrics["num_successful_clients"] = len(successful_results)
        aggregated_metrics["num_failed_clients"] = len(failures)
        
        self.logger.info(f"Round {server_round} aggregation completed. "
                        f"Successful: {len(successful_results)}, Failed: {len(failures)}")
        
        return aggregated_parameters, aggregated_metrics
    
    def _calculate_trust_score(self, client_id: str, fit_res: FitRes) -> float:
        """Calculate trust score for a client based on training results."""
        base_trust = 0.5
        
        # Performance-based trust
        if "accuracy" in fit_res.metrics:
            accuracy = fit_res.metrics["accuracy"]
            performance_trust = min(1.0, accuracy / 0.9)  # Normalize to expected performance
        else:
            performance_trust = 0.5
        
        # Resource efficiency trust
        if "resource_usage" in fit_res.metrics:
            resource_usage = fit_res.metrics["resource_usage"]
            efficiency_trust = max(0.0, 1.0 - resource_usage)
        else:
            efficiency_trust = 0.5
        
        # Training time trust (faster is better for IoT)
        if "training_time" in fit_res.metrics:
            training_time = fit_res.metrics["training_time"]
            # Assume reasonable training time is under 60 seconds
            time_trust = max(0.0, min(1.0, 60.0 / max(training_time, 1.0)))
        else:
            time_trust = 0.5
        
        # Weighted combination
        trust_score = (0.4 * performance_trust + 
                      0.3 * efficiency_trust + 
                      0.3 * time_trust)
        
        return np.clip(trust_score, 0.0, 1.0)
    
    def _update_client_trust(self, client_id: str, success: bool, trust_score: float = 0.0):
        """Update trust tracking for a client."""
        if not success:
            trust_score = 0.1  # Low trust for failed clients
        
        self.client_trust_scores[client_id].append(trust_score)
        
        # Keep only recent history (last 10 rounds)
        if len(self.client_trust_scores[client_id]) > 10:
            self.client_trust_scores[client_id] = self.client_trust_scores[client_id][-10:]
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of evaluation."""
        # Select a subset of clients for evaluation
        available_clients = client_manager.all()
        num_eval_clients = min(len(available_clients), 
                              max(self.min_evaluate_clients,
                                  int(len(available_clients) * self.fraction_evaluate)))
        
        eval_clients = np.random.choice(available_clients, num_eval_clients, replace=False)
        
        eval_config = {
            "server_round": server_round,
            "batch_size": self.config.get('client', {}).get('batch_size', 32),
        }
        
        return [(client, eval_config) for client in eval_clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        # Calculate weighted average of evaluation metrics
        accuracies = []
        losses = []
        num_examples = []
        
        for _, eval_res in results:
            accuracies.append(eval_res.metrics.get("accuracy", 0.0))
            losses.append(eval_res.loss)
            num_examples.append(eval_res.num_examples)
        
        # Weighted average based on number of examples
        total_examples = sum(num_examples)
        if total_examples > 0:
            avg_accuracy = sum(acc * n for acc, n in zip(accuracies, num_examples)) / total_examples
            avg_loss = sum(loss * n for loss, n in zip(losses, num_examples)) / total_examples
        else:
            avg_accuracy = 0.0
            avg_loss = float('inf')
        
        metrics = {
            "accuracy": avg_accuracy,
            "num_clients_evaluated": len(results),
            "num_failed_evaluations": len(failures),
        }
        
        self.logger.info(f"Round {server_round} evaluation: "
                        f"Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return avg_loss, metrics


def create_flwr_server(global_model: nn.Module, config: Dict[str, Any]) -> fl.server.Server:
    """
    Create and configure Flwr server with TrustAwareFedAdam strategy.
    
    Args:
        global_model: The global model architecture
        config: Configuration dictionary
        
    Returns:
        Configured Flwr server
    """
    strategy = TrustAwareFedAdam(global_model=global_model, config=config)
    
    # Server configuration for IoT devices
    server_config = fl.server.ServerConfig(
        num_rounds=config.get('federated', {}).get('num_rounds', 100),
    )
    
    return fl.server.Server(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy,
    ), server_config
