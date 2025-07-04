"""
Refactored main entry point for TRUST-MCNet federated learning training.

This module uses the new SOLID-compliant architecture with:
- Interface-based design and dependency injection
- Registry patterns for extensibility
- Experiment management with comprehensive orchestration
- Production-grade error handling and logging
- Scalable configuration management

Features:
- Uses the new ExperimentManager for orchestration
- Supports 1-1000+ clients via configuration
- Flexible dataset, model, and strategy selection via registries
- Comprehensive metrics and result management
- Robust error handling with custom exceptions

Usage:
    python train_refactored.py                              # Use new architecture
    python train_refactored.py dataset=custom_csv           # Override dataset
    python train_refactored.py model=lstm                   # Use LSTM model
    python train_refactored.py strategy=fedadam             # Use FedAdam strategy
    python train_refactored.py trust=cosine                 # Use cosine similarity trust
    python train_refactored.py experiment.name=my_exp       # Custom experiment name
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our new architecture modules
from core.exceptions import (
    ConfigurationError,
    ExperimentError,
    DataLoadingError,
    ModelError,
    TrustEvaluationError
)
from core.types import ExperimentPhase, ConfigType

# Import registries - using direct imports to avoid relative import issues
import experiments
import data
import models_new
import partitioning
import strategies
import trust_new
import metrics
import clients_new

# Access registries and functions through modules
ExperimentRegistry = experiments.ExperimentRegistry
create_experiment_manager = experiments.create_experiment_manager
DataLoaderRegistry = data.DataLoaderRegistry
ModelRegistry = models_new.ModelRegistry
PartitionerRegistry = partitioning.PartitioningRegistry

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main function for TRUST-MCNet federated learning using new architecture.
    
    This function orchestrates the entire federated learning process using:
    1. Interface-based design with dependency injection
    2. Registry patterns for component selection
    3. Experiment management for comprehensive orchestration
    4. Production-grade error handling and logging
    
    Args:
        cfg: Hydra DictConfig containing all configuration parameters
        
    Returns:
        Dictionary containing experiment results and metrics
        
    Raises:
        ConfigurationError: If configuration is invalid
        ExperimentError: If experiment setup or execution fails
        DataLoadingError: If data loading fails
        ModelError: If model creation or training fails
        TrustEvaluationError: If trust evaluation fails
    """
    experiment_manager = None
    
    try:
        # Convert DictConfig to regular dict for compatibility
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Setup logging
        _setup_logging(config_dict)
        
        logger.info("=" * 80)
        logger.info("TRUST-MCNet Federated Learning Training (Refactored Architecture)")
        logger.info("=" * 80)
        
        # Log configuration summary
        _log_configuration_summary(config_dict)
        
        # Validate configuration
        logger.info("Validating configuration...")
        _validate_configuration(config_dict)
        
        # Create experiment manager using the new architecture
        logger.info("Creating experiment manager...")
        experiment_manager = create_experiment_manager(
            config=config_dict,
            experiment_type="federated_learning"
        )
        
        # Setup experiment
        logger.info("Setting up experiment...")
        experiment_manager.setup()
        
        # Run the experiment
        logger.info("Running federated learning experiment...")
        results = experiment_manager.run()
        
        # Log final results
        _log_experiment_results(results)
        
        logger.info("=" * 80)
        logger.info("TRUST-MCNet Training Completed Successfully")
        logger.info("=" * 80)
        
        return results
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except (DataLoadingError, ModelError, TrustEvaluationError) as e:
        logger.error(f"Component error: {e}")
        raise
    except ExperimentError as e:
        logger.error(f"Experiment error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        logger.error("Traceback:", exc_info=True)
        raise ExperimentError(f"Training failed: {e}") from e
    finally:
        # Ensure cleanup happens
        if experiment_manager:
            try:
                logger.info("Cleaning up experiment resources...")
                experiment_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")


def _setup_logging(config: ConfigType) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = config.get('logging', {}).get('format', 
                                             '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('train_refactored.log')
        ]
    )


def _validate_configuration(config: ConfigType) -> None:
    """
    Validate the configuration for required fields and constraints.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_sections = ['dataset', 'model', 'strategy', 'trust', 'env', 'federated']
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate dataset configuration
    dataset_config = config['dataset']
    if 'name' not in dataset_config:
        raise ConfigurationError("Dataset name not specified")
    if 'num_clients' not in dataset_config or dataset_config['num_clients'] < 1:
        raise ConfigurationError("Invalid number of clients specified")
    
    # Validate model configuration
    model_config = config['model']
    if 'type' not in model_config:
        raise ConfigurationError("Model type not specified")
    
    # Validate federated learning configuration
    federated_config = config['federated']
    if 'num_rounds' not in federated_config or federated_config['num_rounds'] < 1:
        raise ConfigurationError("Invalid number of federated rounds specified")
    
    # Validate that selected components are available in registries
    _validate_component_availability(config)


def _validate_component_availability(config: ConfigType) -> None:
    """
    Validate that configured components are available in registries.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigurationError: If configured component is not available
    """
    # Check data loader availability
    dataset_name = config['dataset']['name']
    if dataset_name not in DataLoaderRegistry.list_available():
        available = ', '.join(DataLoaderRegistry.list_available())
        raise ConfigurationError(
            f"Dataset '{dataset_name}' not available. Available: {available}"
        )
    
    # Check model availability
    model_type = config['model']['type']
    if model_type not in ModelRegistry.list_available():
        available = ', '.join(ModelRegistry.list_available())
        raise ConfigurationError(
            f"Model '{model_type}' not available. Available: {available}"
        )
    
    # Check partitioner availability
    partitioning_strategy = config['dataset'].get('partitioning_strategy', 'iid')
    if partitioning_strategy not in PartitionerRegistry.list_available():
        available = ', '.join(PartitionerRegistry.list_available())
        raise ConfigurationError(
            f"Partitioning strategy '{partitioning_strategy}' not available. Available: {available}"
        )


def _log_configuration_summary(config: ConfigType) -> None:
    """
    Log a summary of the current configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Configuration Summary:")
    logger.info(f"  Dataset: {config['dataset']['name']}")
    logger.info(f"  Model: {config['model']['type']}")
    logger.info(f"  Strategy: {config['strategy']['name']}")
    logger.info(f"  Trust Mode: {config['trust']['mode']}")
    logger.info(f"  Environment: {config['env']['name']}")
    logger.info(f"  Clients: {config['dataset']['num_clients']}")
    logger.info(f"  Rounds: {config['federated']['num_rounds']}")
    logger.info(f"  Batch Size: {config['dataset']['batch_size']}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    
    # Log partitioning strategy
    partitioning = config['dataset'].get('partitioning_strategy', 'iid')
    logger.info(f"  Partitioning: {partitioning}")
    
    # Log Ray configuration
    ray_config = config['env']['ray']
    logger.info(f"  Ray CPUs: {ray_config['num_cpus']}")
    logger.info(f"  Ray GPUs: {ray_config['num_gpus']}")
    logger.info(f"  Ray Memory: {ray_config['object_store_memory'] / 1e9:.1f}GB")
    
    # Log available components
    logger.info("Available Components:")
    logger.info(f"  Data Loaders: {', '.join(DataLoaderRegistry.list_available())}")
    logger.info(f"  Models: {', '.join(ModelRegistry.list_available())}")
    logger.info(f"  Partitioners: {', '.join(PartitionerRegistry.list_available())}")


def _log_experiment_results(results: Dict[str, Any]) -> None:
    """
    Log experiment results and metrics.
    
    Args:
        results: Experiment results dictionary
    """
    logger.info("Experiment Results:")
    
    # Basic experiment info
    basic_info = ['experiment_name', 'simulation_time', 'num_clients', 'num_rounds', 
                  'dataset_size', 'strategy', 'trust_mode']
    
    for key in basic_info:
        if key in results:
            value = results[key]
            if isinstance(value, float):
                logger.info(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Log client split statistics
    if 'client_split_sizes' in results:
        split_sizes = results['client_split_sizes']
        logger.info("  Client Split Statistics:")
        logger.info(f"    Min: {min(split_sizes)}")
        logger.info(f"    Max: {max(split_sizes)}")
        logger.info(f"    Avg: {sum(split_sizes) / len(split_sizes):.1f}")
        logger.info(f"    Std: {np.std(split_sizes):.1f}")
    
    # Log final metrics if available
    if 'final_metrics' in results:
        final_metrics = results['final_metrics']
        logger.info("  Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.4f}")
            else:
                logger.info(f"    {key}: {value}")
    
    # Log training history summary
    if 'history' in results:
        history = results['history']
        logger.info("  Training History:")
        
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            logger.info(f"    Total Rounds: {len(history.losses_distributed)}")
            final_loss = history.losses_distributed[-1][1]
            logger.info(f"    Final Loss: {final_loss:.4f}")
        
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            final_round_metrics = history.metrics_distributed[-1][1]
            for metric_name, metric_values in final_round_metrics.items():
                if metric_values:
                    avg_metric = sum(metric_values) / len(metric_values)
                    logger.info(f"    Final Avg {metric_name}: {avg_metric:.4f}")
    
    # Log resource usage if available
    if 'resource_usage' in results:
        resource_usage = results['resource_usage']
        logger.info("  Resource Usage:")
        for key, value in resource_usage.items():
            logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    """
    Entry point for the refactored TRUST-MCNet training.
    
    This uses the new SOLID-compliant architecture with:
    - Interface-based design
    - Registry patterns for extensibility
    - Comprehensive experiment management
    - Production-grade error handling
    
    Examples:
        # Basic training with new architecture
        python train_refactored.py
        
        # Override specific configurations
        python train_refactored.py dataset=mnist model=mlp strategy=fedavg
        
        # Run with different trust mechanisms
        python train_refactored.py trust=cosine
        
        # Scale to many clients
        python train_refactored.py dataset.num_clients=100
        
        # Custom experiment name
        python train_refactored.py experiment.name=large_scale_test
    """
    import numpy as np  # Import here to avoid issues with logging setup
    main()
