"""
Enhanced simulation orchestrator for TRUST-MCNet federated learning.

This module orchestrates the federated learning simulation using:
- Ray for distributed client execution and resource management
- Flower for federated learning coordination
- Trust mechanisms for client evaluation and selection
- Enhanced data partitioning strategies
- Comprehensive metrics logging
- Proper resource cleanup

Key responsibilities:
- Initialize Ray with proper cleanup
- Load datasets using registry pattern
- Partition data using strategy pattern
- Create enhanced Ray actor clients
- Configure Flower strategy with trust mechanisms
- Run federated learning simulation with metrics logging
- Cleanup resources properly
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import ray
import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg, FedAdam, FedProx
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns
import torch
import torch.nn as nn

# Import our enhanced modules
from utils.dataset_registry import DataManager
from utils.partitioning import PartitionerRegistry
from utils.ray_utils import ray_context, cleanup_training_resources, cleanup_evaluation_resources
from utils.metrics_logger import create_metrics_manager
from clients.enhanced_ray_client import create_ray_client_fn
from models.model import MLP, LSTM
from trust_module.trust_evaluator import TrustEvaluator
from config.schemas import Config

logger = logging.getLogger(__name__)


def run_enhanced_simulation(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run enhanced federated learning simulation with comprehensive improvements.
    
    This function orchestrates the entire federated learning process with:
    - Proper Ray resource management
    - Strategy pattern for data partitioning
    - Registry pattern for dataset loading
    - Enhanced error handling and retries
    - Comprehensive metrics logging
    - Memory cleanup
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary containing simulation results and metrics
    """
    logger.info("Starting enhanced TRUST-MCNet federated learning simulation")
    start_time = time.time()
    
    # Initialize metrics manager
    experiment_name = f"federated_experiment_{int(time.time())}"
    metrics_manager = create_metrics_manager(cfg, experiment_name)
    
    try:
        # Use Ray context manager for proper cleanup
        with ray_context(cfg.env.ray) as ray_manager:
            logger.info("Ray initialized successfully")
            
            # Step 1: Load dataset using registry pattern
            logger.info(f"Loading dataset: {cfg.dataset.name}")
            data_manager = DataManager(cfg.dataset)
            
            try:
                train_dataset, test_dataset = data_manager.load_datasets()
                dataset_info = data_manager.get_data_info()
                logger.info(f"Dataset loaded: {len(train_dataset)} samples, "
                           f"shape: {dataset_info['data_shape']}, "
                           f"classes: {dataset_info['num_classes']}")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
            
            # Step 2: Validate dataset and client configuration
            num_clients = cfg.dataset.num_clients
            dataset_size = len(train_dataset)
            
            if num_clients < 1:
                raise ValueError(f"num_clients must be >= 1, got {num_clients}")
            
            if dataset_size < num_clients:
                raise ValueError(
                    f"Dataset size ({dataset_size}) < num_clients ({num_clients}). "
                    f"Each client needs at least 1 sample."
                )
            
            min_samples_per_client = cfg.dataset.get('min_samples_per_client', 1)
            if dataset_size < (num_clients * min_samples_per_client):
                raise ValueError(
                    f"Insufficient data: {dataset_size} samples for {num_clients} clients "
                    f"with minimum {min_samples_per_client} samples per client"
                )
            
            logger.info(f"Dataset validation passed: {dataset_size} samples for {num_clients} clients")
            
            # Step 3: Partition dataset using strategy pattern
            partitioning_strategy = cfg.dataset.get('partitioning', 'iid')
            logger.info(f"Partitioning dataset using {partitioning_strategy} strategy")
            
            try:
                partitioner = PartitionerRegistry.get_partitioner(partitioning_strategy)
                
                # Prepare partitioning parameters
                partition_kwargs = {}
                if partitioning_strategy == 'dirichlet':
                    partition_kwargs['alpha'] = cfg.dataset.get('dirichlet_alpha', 0.5)
                elif partitioning_strategy == 'pathological':
                    partition_kwargs['classes_per_client'] = cfg.dataset.get('classes_per_client', 2)
                
                client_subsets = partitioner.partition(
                    dataset=train_dataset,
                    num_clients=num_clients,
                    **partition_kwargs
                )
                
                # Validate partitions
                partitioner.validate_partition(client_subsets, min_samples_per_client)
                
            except Exception as e:
                logger.error(f"Dataset partitioning failed: {e}")
                raise
            
            # Log partition statistics
            split_sizes = [len(subset) for subset in client_subsets]
            logger.info(f"Client partition sizes: min={min(split_sizes)}, "
                       f"max={max(split_sizes)}, avg={sum(split_sizes)/len(split_sizes):.1f}")
            
            # Step 4: Create enhanced client function
            logger.info("Creating enhanced Ray-based client function")
            try:
                client_fn = create_ray_client_fn(client_subsets, cfg)
            except Exception as e:
                logger.error(f"Failed to create client function: {e}")
                raise
            
            # Step 5: Initialize model and get initial parameters
            logger.info("Initializing global model")
            try:
                model = _create_initial_model(cfg.model, dataset_info)
                initial_parameters = _get_model_parameters(model)
                logger.info(f"Model initialized with {len(initial_parameters)} parameter arrays")
            except Exception as e:
                logger.error(f"Model initialization failed: {e}")
                raise
            
            # Step 6: Configure enhanced Flower strategy
            logger.info("Configuring Flower strategy with trust mechanisms")
            try:
                strategy = _create_enhanced_strategy(cfg, initial_parameters, metrics_manager)
            except Exception as e:
                logger.error(f"Strategy configuration failed: {e}")
                raise
            
            # Step 7: Run federated learning simulation
            logger.info(f"Starting simulation: {num_clients} clients, "
                       f"{cfg.federated.num_rounds} rounds")
            
            simulation_start_time = time.time()
            
            try:
                history = start_simulation(
                    client_fn=client_fn,
                    num_clients=num_clients,
                    config=fl.server.ServerConfig(num_rounds=cfg.federated.num_rounds),
                    strategy=strategy,
                    client_resources=cfg.env.simulation.client_resources,
                    ray_init_args={"ignore_reinit_error": True}  # Ray already initialized
                )
                
                simulation_time = time.time() - simulation_start_time
                logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Simulation execution failed: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Step 8: Collect and analyze results
            total_time = time.time() - start_time
            
            results = {
                'history': history,
                'simulation_time': simulation_time,
                'total_time': total_time,
                'num_clients': num_clients,
                'num_rounds': cfg.federated.num_rounds,
                'dataset_size': dataset_size,
                'dataset_info': dataset_info,
                'client_split_sizes': split_sizes,
                'partitioning_strategy': partitioning_strategy,
                'strategy': cfg.strategy.name,
                'trust_mode': cfg.trust.mode,
                'final_metrics': _extract_final_metrics(history),
                'experiment_name': experiment_name
            }
            
            # Log final metrics
            metrics_manager.log_server_metrics({
                'simulation_time': simulation_time,
                'total_time': total_time,
                'final_loss': results['final_metrics'].get('loss', float('inf')),
                'final_accuracy': results['final_metrics'].get('accuracy', 0.0)
            })
            
            logger.info(f"Enhanced simulation completed successfully")
            logger.info(f"Total time: {total_time:.2f}s, Final metrics: {results['final_metrics']}")
            
            return results
            
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        logger.error(traceback.format_exc())
        
        # Log error metrics
        try:
            metrics_manager.log_server_metrics({
                'error': str(e),
                'simulation_failed': True
            })
        except:
            pass
        
        raise
    
    finally:
        # Always cleanup metrics manager
        try:
            metrics_manager.export_metrics(f"logs/{experiment_name}/final_metrics.json")
            metrics_manager.close()
        except Exception as e:
            logger.warning(f"Failed to close metrics manager: {e}")
        
        # Final cleanup
        cleanup_training_resources()
        cleanup_evaluation_resources()


def _create_initial_model(model_config: DictConfig, dataset_info: Dict[str, Any]) -> nn.Module:
    """
    Create initial model based on configuration and dataset info.
    
    Args:
        model_config: Model configuration
        dataset_info: Dataset information
        
    Returns:
        Initialized PyTorch model
    """
    model_type = model_config.get('type', model_config.get('name', 'MLP'))
    
    # Determine input dimension from dataset
    data_shape = dataset_info['data_shape']
    if len(data_shape) == 3:  # Image data (C, H, W)
        input_dim = data_shape[0] * data_shape[1] * data_shape[2]
    elif len(data_shape) == 1:  # Feature vector
        input_dim = data_shape[0]
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}")
    
    output_dim = dataset_info['num_classes']
    
    if model_type.upper() == 'MLP':
        # MLP now has a fixed architecture
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim
        )
    elif model_type.upper() == 'LSTM':
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 2),
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model: input_dim={input_dim}, output_dim={output_dim}")
    return model


def _get_model_parameters(model: nn.Module) -> List:
    """
    Extract model parameters as list of NumPy arrays.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of model parameters as NumPy arrays
    """
    try:
        parameters = []
        for param in model.parameters():
            parameters.append(param.detach().cpu().numpy())
        return parameters
    except Exception as e:
        logger.error(f"Failed to extract model parameters: {e}")
        raise


def _create_enhanced_strategy(
    cfg: DictConfig, 
    initial_parameters: List, 
    metrics_manager
) -> fl.server.strategy.Strategy:
    """
    Create enhanced Flower strategy with trust mechanisms and metrics logging.
    
    Args:
        cfg: Configuration
        initial_parameters: Initial model parameters
        metrics_manager: Metrics logging manager
        
    Returns:
        Configured Flower strategy
    """
    strategy_name = cfg.strategy.name.lower()
    
    # Initialize trust evaluator
    trust_evaluator = TrustEvaluator(
        trust_mode=cfg.trust.mode,
        threshold=cfg.trust.threshold
    )
    
    # Convert parameters to Flower format
    initial_params = ndarrays_to_parameters(initial_parameters)
    
    # Common strategy arguments
    strategy_args = {
        'fraction_fit': cfg.federated.fraction_fit,
        'fraction_evaluate': cfg.federated.fraction_evaluate,
        'min_fit_clients': cfg.federated.min_fit_clients,
        'min_evaluate_clients': cfg.federated.min_evaluate_clients,
        'min_available_clients': cfg.federated.min_available_clients,
        'initial_parameters': initial_params,
    }
    
    # Create strategy based on configuration
    if strategy_name == 'fedavg':
        strategy = FedAvg(**strategy_args)
    elif strategy_name == 'fedadam':
        strategy_args.update({
            'eta': cfg.strategy.get('eta', 1e-3),
            'eta_l': cfg.strategy.get('eta_l', 1e-3),
            'beta_1': cfg.strategy.get('beta_1', 0.9),
            'beta_2': cfg.strategy.get('beta_2', 0.99),
            'tau': cfg.strategy.get('tau', 1e-9)
        })
        strategy = FedAdam(**strategy_args)
    elif strategy_name == 'fedprox':
        strategy_args.update({
            'proximal_mu': cfg.strategy.get('proximal_mu', 1.0)
        })
        strategy = FedProx(**strategy_args)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Wrap strategy with trust mechanisms and metrics logging
    enhanced_strategy = EnhancedStrategyWrapper(
        base_strategy=strategy,
        trust_evaluator=trust_evaluator,
        metrics_manager=metrics_manager,
        cfg=cfg
    )
    
    return enhanced_strategy


def _extract_final_metrics(history) -> Dict[str, Any]:
    """
    Extract final metrics from Flower history.
    
    Args:
        history: Flower simulation history
        
    Returns:
        Dictionary of final metrics
    """
    try:
        final_metrics = {}
        
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            final_metrics['loss'] = final_loss
        
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            final_round_metrics = history.metrics_distributed[-1][1]
            if 'accuracy' in final_round_metrics:
                final_metrics['accuracy'] = final_round_metrics['accuracy']
        
        return final_metrics
        
    except Exception as e:
        logger.warning(f"Failed to extract final metrics: {e}")
        return {}


class EnhancedStrategyWrapper(fl.server.strategy.Strategy):
    """
    Enhanced strategy wrapper with trust mechanisms and metrics logging.
    """
    
    def __init__(
        self, 
        base_strategy: fl.server.strategy.Strategy,
        trust_evaluator: TrustEvaluator,
        metrics_manager,
        cfg: DictConfig
    ):
        """
        Initialize enhanced strategy wrapper.
        
        Args:
            base_strategy: Base Flower strategy
            trust_evaluator: Trust evaluation system
            metrics_manager: Metrics logging manager
            cfg: Configuration
        """
        self.base_strategy = base_strategy
        self.trust_evaluator = trust_evaluator
        self.metrics_manager = metrics_manager
        self.cfg = cfg
        self.current_round = 0
        
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        return self.base_strategy.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training with trust-based selection."""
        self.current_round = server_round
        self.metrics_manager.start_round(server_round)
        
        # Get base strategy configuration
        config = self.base_strategy.configure_fit(server_round, parameters, client_manager)
        
        # Add training configuration to each client's config
        training_config = {
            'epochs': self.cfg.training.epochs,
            'learning_rate': self.cfg.training.learning_rate,
            'optimizer': self.cfg.training.optimizer
        }
        
        # Update client configurations with training parameters
        updated_config = []
        for client_proxy, fit_ins in config:
            # Merge training config with existing config
            updated_fit_config = dict(fit_ins.config)
            updated_fit_config.update(training_config)
            
            # Create new FitIns with updated config
            updated_fit_ins = FitIns(fit_ins.parameters, updated_fit_config)
            updated_config.append((client_proxy, updated_fit_ins))
        
        return updated_config
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure clients for evaluation."""
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results with trust evaluation."""
        try:
            # Log client training metrics
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                if hasattr(fit_res, 'metrics') and fit_res.metrics:
                    self.metrics_manager.log_client_metrics(client_id, fit_res.metrics)
            
            # Perform base aggregation
            aggregated_result = self.base_strategy.aggregate_fit(server_round, results, failures)
            
            # Evaluate trust if enabled
            if self.cfg.trust.mode != 'none':
                try:
                    # Extract client updates for trust evaluation
                    client_updates = {}
                    for client_proxy, fit_res in results:
                        client_id = client_proxy.cid
                        client_updates[client_id] = parameters_to_ndarrays(fit_res.parameters)
                    
                    # Evaluate trust scores
                    trust_scores = self.trust_evaluator.evaluate_trust_batch(client_updates)
                    
                    # Log trust metrics
                    self.metrics_manager.log_trust_metrics(trust_scores)
                    
                except Exception as e:
                    logger.warning(f"Trust evaluation failed: {e}")
            
            # Log aggregation metrics
            self.metrics_manager.log_aggregation_metrics({
                'num_clients_fit': len(results),
                'num_failures': len(failures),
                'aggregation_successful': aggregated_result is not None
            })
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Fit aggregation failed: {e}")
            raise
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        try:
            # Log client evaluation metrics
            for client_proxy, evaluate_res in results:
                client_id = client_proxy.cid
                metrics = {
                    'loss': evaluate_res.loss,
                    'num_examples': evaluate_res.num_examples
                }
                if hasattr(evaluate_res, 'metrics') and evaluate_res.metrics:
                    metrics.update(evaluate_res.metrics)
                
                self.metrics_manager.log_client_metrics(f"{client_id}_eval", metrics)
            
            # Perform base aggregation
            aggregated_result = self.base_strategy.aggregate_evaluate(server_round, results, failures)
            
            # Log server evaluation metrics
            if aggregated_result is not None:
                server_metrics = {
                    'server_loss': aggregated_result[0],
                    'server_metrics': aggregated_result[1] if len(aggregated_result) > 1 else {}
                }
                self.metrics_manager.log_server_metrics(server_metrics)
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Evaluate aggregation failed: {e}")
            raise
    
    def evaluate(self, server_round, parameters):
        """Evaluate global model."""
        return self.base_strategy.evaluate(server_round, parameters)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for enhanced federated learning simulation.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format
    )
    
    logger.info("Starting TRUST-MCNet Enhanced Federated Learning")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Run enhanced simulation
        results = run_enhanced_simulation(cfg)
        
        logger.info("Simulation completed successfully!")
        logger.info(f"Final results: {results['final_metrics']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
