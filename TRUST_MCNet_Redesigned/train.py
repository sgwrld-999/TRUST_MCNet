"""
Main entry point for TRUST-MCNet federated learning training.

This module serves as the Hydra entry point for the federated learning simulation.
It loads configuration, validates settings, and orchestrates the training process
using Ray and Flower frameworks with trust mechanisms.

Features:
- Enhanced simulation with new architecture patterns (default)
- Legacy simulation for backward compatibility
- Hydra configuration management with OmegaConf schemas
- Comprehensive error handling and logging

Usage:
    python train.py                              # Use enhanced simulation (default)
    python train.py simulation.use_legacy=true   # Use legacy simulation
    python train.py dataset=custom_csv           # Override dataset
    python train.py env=gpu                      # Use GPU environment
    python train.py strategy=fedadam             # Use FedAdam strategy
    python train.py trust=cosine                 # Use cosine similarity trust
    
Hydra allows easy configuration overrides and experiment management.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import both simulation modules
from simulation import run_simulation, setup_logging, validate_configuration
from enhanced_simulation import run_enhanced_simulation

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main function for TRUST-MCNet federated learning training.
    
    This function serves as the Hydra entry point and orchestrates the entire
    federated learning process using either the enhanced simulation (default)
    or legacy simulation for backward compatibility.
    
    Process:
    1. Setup logging and configuration validation
    2. Convert Hydra DictConfig for compatibility
    3. Choose simulation type based on configuration
    4. Initialize and run the federated learning simulation
    5. Handle results and cleanup
    
    Args:
        cfg: Hydra DictConfig containing all configuration parameters
        
    Returns:
        Dictionary containing simulation results and metrics
        
    Raises:
        Exception: If simulation fails or configuration is invalid
    """
    try:
        # Convert DictConfig to regular dict for easier handling
        # OmegaConf.to_container resolves interpolations and converts to Python objects
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Setup logging based on configuration
        setup_logging(config_dict)
        
        logger.info("=" * 60)
        logger.info("TRUST-MCNet Federated Learning Training Started")
        logger.info("=" * 60)
        
        # Log configuration summary
        _log_configuration_summary(config_dict)
        
        # Validate configuration before proceeding
        logger.info("Validating configuration...")
        validate_configuration(config_dict)
        
        # Determine which simulation to use
        use_legacy = config_dict.get('simulation', {}).get('use_legacy', False)
        
        if use_legacy:
            logger.info("Using legacy simulation for backward compatibility...")
            results = run_simulation(config_dict)
        else:
            logger.info("Using enhanced simulation with new architecture patterns...")
            results = run_enhanced_simulation(cfg)  # Pass original DictConfig for enhanced simulation
        
        # Log final results
        _log_simulation_results(results)
        
        logger.info("=" * 60)
        logger.info("TRUST-MCNet Training Completed Successfully")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Traceback:", exc_info=True)
        raise


def _log_configuration_summary(cfg: Dict[str, Any]) -> None:
    """
    Log a summary of the current configuration.
    
    Args:
        cfg: Configuration dictionary
    """
    logger.info("Configuration Summary:")
    logger.info(f"  Dataset: {cfg['dataset']['name']}")
    logger.info(f"  Model: {cfg['model']['type']}")
    logger.info(f"  Strategy: {cfg['strategy']['name']}")
    logger.info(f"  Trust Mode: {cfg['trust']['mode']}")
    logger.info(f"  Environment: {cfg['env']['name']}")
    logger.info(f"  Clients: {cfg['dataset']['num_clients']}")
    logger.info(f"  Rounds: {cfg['federated']['num_rounds']}")
    logger.info(f"  Batch Size: {cfg['dataset']['batch_size']}")
    logger.info(f"  Learning Rate: {cfg['training']['learning_rate']}")
    
    # Log Ray configuration
    ray_config = cfg['env']['ray']
    logger.info(f"  Ray CPUs: {ray_config['num_cpus']}")
    logger.info(f"  Ray GPUs: {ray_config['num_gpus']}")
    logger.info(f"  Ray Memory: {ray_config['object_store_memory'] / 1e9:.1f}GB")


def _log_simulation_results(results: Dict[str, Any]) -> None:
    """
    Log simulation results and metrics.
    
    Args:
        results: Simulation results dictionary
    """
    logger.info("Simulation Results:")
    logger.info(f"  Simulation Time: {results['simulation_time']:.2f} seconds")
    logger.info(f"  Total Clients: {results['num_clients']}")
    logger.info(f"  Total Rounds: {results['num_rounds']}")
    logger.info(f"  Dataset Size: {results['dataset_size']}")
    logger.info(f"  Strategy Used: {results['strategy']}")
    logger.info(f"  Trust Mode: {results['trust_mode']}")
    
    # Log client split statistics
    split_sizes = results['client_split_sizes']
    logger.info(f"  Client Split Sizes:")
    logger.info(f"    Min: {min(split_sizes)}")
    logger.info(f"    Max: {max(split_sizes)}")
    logger.info(f"    Avg: {sum(split_sizes) / len(split_sizes):.1f}")
    
    # Log final metrics if available
    final_metrics = results.get('final_metrics', {})
    if final_metrics:
        logger.info("  Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.4f}")
            else:
                logger.info(f"    {key}: {value}")
    
    # Log history summary
    history = results.get('history')
    if history:
        logger.info(f"  Training History:")
        logger.info(f"    Distributed Rounds: {len(history.losses_distributed)}")
        if history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            logger.info(f"    Final Distributed Loss: {final_loss:.4f}")
        
        if history.metrics_distributed:
            final_round_metrics = history.metrics_distributed[-1][1]
            for metric_name, metric_values in final_round_metrics.items():
                if metric_values:
                    avg_metric = sum(metric_values) / len(metric_values)
                    logger.info(f"    Final Avg {metric_name}: {avg_metric:.4f}")


if __name__ == "__main__":
    """
    Entry point for the application.
    
    When this script is run directly, it will:
    1. Initialize Hydra with the configuration
    2. Load and validate all configuration files
    3. Execute the main training function
    4. Handle any errors and provide meaningful feedback
    
    Hydra features available:
    - Configuration composition and overrides
    - Experiment tracking and output management
    - Multirun sweeps for hyperparameter optimization
    - Working directory management
    
    Examples:
        # Basic training with default config
        python train.py
        
        # Override specific configurations
        python train.py dataset=custom_csv strategy=fedadam
        
        # Run hyperparameter sweep
        python train.py -m strategy=fedavg,fedadam trust=cosine,entropy
        
        # Change output directory
        python train.py hydra.run.dir=./outputs/my_experiment
    """
    main()
