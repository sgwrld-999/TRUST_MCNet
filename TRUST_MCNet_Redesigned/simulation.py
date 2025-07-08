"""
Simulation orchestrator for TRUST-MCNet federated learning.

This module orchestrates the federated learning simulation using:
- Ray for distributed client execution and resource management
- Flower for federated learning coordination
- Trust mechanisms for client evaluation and selection

Key responsibilities:
- Initialize Ray with configuration
- Load and split datasets among clients
- Create Ray actor clients
- Configure Flower strategy
- Run federated learning simulation
"""

import logging
import time
from typing import Dict, List, Any, Optional
import ray
import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg, FedAdam, FedProx
from flwr.common import ndarrays_to_parameters
import torch
import torch.nn as nn

from utils.data_utils import (
    CSVDataset, 
    load_mnist_dataset, 
    split_clients, 
    validate_data_splits
)
from clients.ray_flwr_client import create_ray_client_fn
from models.model import MLP, LSTM
from trust_module.trust_evaluator import TrustEvaluator

logger = logging.getLogger(__name__)


def run_simulation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run federated learning simulation with Ray and Flower.
    
    This function orchestrates the entire federated learning process:
    1. Initialize Ray with resource configuration
    2. Load and validate dataset
    3. Split data among clients with edge case handling
    4. Create Ray actor clients
    5. Configure Flower strategy with trust mechanisms
    6. Execute federated learning simulation
    7. Collect and return results
    
    Args:
        cfg: Complete Hydra configuration
        
    Returns:
        Dictionary containing simulation results and metrics
    """
    logger.info("Starting TRUST-MCNet federated learning simulation")
    start_time = time.time()
    
    try:
        # Step 1: Initialize Ray with configuration
        ray_config = cfg['env']['ray']
        logger.info(f"Initializing Ray with config: {ray_config}")
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=ray_config.get('num_cpus', 4),
                num_gpus=ray_config.get('num_gpus', 0),
                object_store_memory=ray_config.get('object_store_memory', 1000000000),
                dashboard_host=ray_config.get('dashboard_host', "127.0.0.1"),
                dashboard_port=ray_config.get('dashboard_port', 8265),
                ignore_reinit_error=ray_config.get('ignore_reinit_error', True)
            )
        
        # Step 2: Load dataset based on configuration
        dataset_config = cfg['dataset']
        logger.info(f"Loading dataset: {dataset_config['name']}")
        
        if dataset_config['name'] == 'mnist':
            train_dataset, test_dataset = load_mnist_dataset(
                data_path=dataset_config['path'],
                binary_classification=dataset_config.get('binary_classification')
            )
            # Use train dataset for federated learning
            full_dataset = train_dataset
            
        elif dataset_config['name'] == 'custom_csv':
            full_dataset = CSVDataset(
                csv_path=dataset_config['path'],
                target_column=dataset_config['csv']['target_column'],
                feature_columns=dataset_config['csv']['feature_columns'],
                preprocessing=dataset_config.get('preprocessing', {})
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_config['name']}")
        
        # Step 3: Validate dataset and client configuration
        num_clients = dataset_config['num_clients']
        dataset_size = len(full_dataset)
        
        # Edge case validation: Ensure feasible configuration
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")
        
        if dataset_size < num_clients:
            raise ValueError(
                f"Dataset size ({dataset_size}) < num_clients ({num_clients}). "
                f"Each client needs at least 1 sample."
            )
        
        min_samples_per_client = dataset_config.get('min_samples_per_client', 1)
        if dataset_size < (num_clients * min_samples_per_client):
            raise ValueError(
                f"Insufficient data: {dataset_size} samples for {num_clients} clients "
                f"with minimum {min_samples_per_client} samples per client"
            )
        
        logger.info(f"Dataset validation passed: {dataset_size} samples for {num_clients} clients")
        
        # Step 4: Split dataset among clients
        logger.info(f"Splitting dataset using {dataset_config.get('partitioning', 'iid')} partitioning")
        
        client_subsets = split_clients(
            dataset=full_dataset,
            num_clients=num_clients,
            partitioning=dataset_config.get('partitioning', 'iid'),
            dirichlet_alpha=dataset_config.get('dirichlet_alpha', 0.5)
        )
        
        # Validate client splits
        validate_data_splits(
            client_subsets=client_subsets,
            min_samples_per_client=min_samples_per_client,
            max_samples_per_client=dataset_config.get('max_samples_per_client')
        )
        
        # Log client split statistics
        split_sizes = [len(subset) for subset in client_subsets]
        logger.info(f"Client split sizes: min={min(split_sizes)}, "
                   f"max={max(split_sizes)}, avg={sum(split_sizes)/len(split_sizes):.1f}")
        
        # Step 5: Create client function for Flower simulation
        logger.info("Creating Ray-based client function")
        client_fn = create_ray_client_fn(client_subsets, cfg)
        
        # Step 6: Initialize model and get initial parameters
        model = _create_initial_model(cfg['model'])
        initial_parameters = _get_model_parameters(model)
        
        # Step 7: Configure Flower strategy
        strategy = _create_strategy(cfg, initial_parameters)
        
        # Step 8: Configure simulation parameters
        simulation_config = {
            'num_clients': num_clients,
            'num_rounds': cfg['federated']['num_rounds'],
            'client_resources': cfg['env']['simulation']['client_resources'],
            'ray_init_args': {"ignore_reinit_error": True}  # Ray already initialized
        }
        
        logger.info(f"Starting simulation with {num_clients} clients for "
                   f"{cfg['federated']['num_rounds']} rounds")
        
        # Step 9: Run federated learning simulation
        history = start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg['federated']['num_rounds']),
            strategy=strategy,
            client_resources=simulation_config['client_resources'],
            ray_init_args=simulation_config['ray_init_args']
        )
        
        # Step 10: Collect results
        simulation_time = time.time() - start_time
        
        results = {
            'history': history,
            'simulation_time': simulation_time,
            'num_clients': num_clients,
            'num_rounds': cfg['federated']['num_rounds'],
            'dataset_size': dataset_size,
            'client_split_sizes': split_sizes,
            'strategy': cfg['strategy']['name'],
            'trust_mode': cfg['trust']['mode'],
            'final_metrics': _extract_final_metrics(history)
        }
        
        logger.info(f"Simulation completed successfully in {simulation_time:.2f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    
    finally:
        # Cleanup Ray resources
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown completed")


def _create_initial_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create initial model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Initialized PyTorch model
    """
    if model_config['type'] == 'MLP':
        model = MLP(
            input_dim=model_config['mlp']['input_dim'],
            output_dim=model_config['mlp']['output_dim']
        )
    elif model_config['type'] == 'LSTM':
        model = LSTM(
            input_dim=model_config['lstm']['input_dim'],
            hidden_dim=model_config['lstm']['hidden_dim'],
            num_layers=model_config['lstm']['num_layers'],
            output_dim=model_config['lstm']['output_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    logger.info(f"Created {model_config['type']} model with "
               f"{sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def _get_model_parameters(model: nn.Module) -> fl.common.Parameters:
    """
    Extract model parameters for Flower.
    
    Args:
        model: PyTorch model
        
    Returns:
        Flower Parameters object
    """
    # Convert model parameters to numpy arrays
    parameters = []
    for param in model.parameters():
        parameters.append(param.detach().cpu().numpy())
    
    # Convert to Flower Parameters
    return ndarrays_to_parameters(parameters)


def _create_strategy(cfg: Dict[str, Any], initial_parameters: fl.common.Parameters) -> fl.server.strategy.Strategy:
    """
    Create Flower strategy based on configuration.
    
    Args:
        cfg: Complete configuration
        initial_parameters: Initial model parameters
        
    Returns:
        Configured Flower strategy
    """
    strategy_config = cfg['strategy']
    strategy_name = strategy_config['name'].lower()
    
    # Common strategy parameters
    common_params = {
        'fraction_fit': strategy_config.get('fraction_fit', cfg['federated']['fraction_fit']),
        'fraction_evaluate': strategy_config.get('fraction_evaluate', cfg['federated']['fraction_evaluate']),
        'min_fit_clients': strategy_config.get('min_fit_clients', cfg['federated']['min_fit_clients']),
        'min_evaluate_clients': strategy_config.get('min_evaluate_clients', cfg['federated']['min_evaluate_clients']),
        'min_available_clients': strategy_config.get('min_available_clients', cfg['federated']['min_available_clients']),
        'initial_parameters': initial_parameters,
        'accept_failures': strategy_config.get('accept_failures', True)
    }
    
    # Create strategy based on configuration
    if strategy_name == 'fedavg':
        strategy = FedAvg(**common_params)
        
    elif strategy_name == 'fedadam':
        fedadam_params = common_params.copy()
        fedadam_params.update({
            'eta': strategy_config.get('eta', 0.01),
            'eta_l': strategy_config.get('eta_l', 0.01),
            'beta_1': strategy_config.get('beta_1', 0.9),
            'beta_2': strategy_config.get('beta_2', 0.999),
            'tau': strategy_config.get('tau', 0.001)
        })
        strategy = FedAdam(**fedadam_params)
        
    elif strategy_name == 'fedprox':
        fedprox_params = common_params.copy()
        fedprox_params.update({
            'proximal_mu': strategy_config.get('proximal_mu', 0.01)
        })
        strategy = FedProx(**fedprox_params)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    logger.info(f"Created {strategy_name} strategy with parameters: {strategy_config}")
    return strategy


def _extract_final_metrics(history: fl.server.history.History) -> Dict[str, Any]:
    """
    Extract final metrics from simulation history.
    
    Args:
        history: Flower simulation history
        
    Returns:
        Dictionary of final metrics
    """
    try:
        final_metrics = {}
        
        # Extract final round losses
        if history.losses_distributed:
            final_metrics['final_loss'] = history.losses_distributed[-1][1]
        
        # Extract final round metrics
        if history.metrics_distributed:
            final_round_metrics = history.metrics_distributed[-1][1]
            for key, values in final_round_metrics.items():
                # Calculate average across clients
                if values:
                    final_metrics[f'final_{key}'] = sum(values) / len(values)
        
        # Extract centralized metrics if available
        if history.losses_centralized:
            final_metrics['final_centralized_loss'] = history.losses_centralized[-1][1]
        
        if history.metrics_centralized:
            final_centralized = history.metrics_centralized[-1][1]
            for key, value in final_centralized.items():
                final_metrics[f'final_centralized_{key}'] = value
        
        return final_metrics
        
    except Exception as e:
        logger.warning(f"Error extracting final metrics: {e}")
        return {}


def setup_logging(cfg: Dict[str, Any]) -> None:
    """
    Setup logging configuration.
    
    Args:
        cfg: Configuration dictionary
    """
    log_level = cfg.get('logging', {}).get('level', 'INFO')
    log_format = cfg.get('logging', {}).get('format', 
                                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('ray').setLevel(logging.WARNING)
    logging.getLogger('flwr').setLevel(logging.INFO)


def validate_configuration(cfg: Dict[str, Any]) -> None:
    """
    Validate configuration before starting simulation.
    
    Args:
        cfg: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required sections
    required_sections = ['dataset', 'env', 'strategy', 'trust', 'model', 'federated', 'training']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate federated learning parameters
    federated_config = cfg['federated']
    if federated_config['num_rounds'] < 1:
        raise ValueError("num_rounds must be >= 1")
    
    if not (0 < federated_config['fraction_fit'] <= 1):
        raise ValueError("fraction_fit must be in (0, 1]")
    
    if not (0 < federated_config['fraction_evaluate'] <= 1):
        raise ValueError("fraction_evaluate must be in (0, 1]")
    
    # Validate dataset configuration
    dataset_config = cfg['dataset']
    if dataset_config['num_clients'] < 1:
        raise ValueError("num_clients must be >= 1")
    
    if not (0 < dataset_config['eval_fraction'] < 1):
        raise ValueError("eval_fraction must be in (0, 1)")
    
    # Validate model configuration
    model_config = cfg['model']
    if model_config['type'] not in ['MLP', 'LSTM']:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    logger.info("Configuration validation passed")


if __name__ == "__main__":
    # This module should not be run directly
    # Use train.py as the entry point
    print("This module should not be run directly. Use train.py as the entry point.")
