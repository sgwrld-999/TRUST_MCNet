#!/usr/bin/env python3
"""
Flower Server Launcher with Trust-Weighted Aggregation

This script demonstrates how to launch a Flower server using the TrustWeightedStrategy
following the implementation guide. It integrates the TRUST_MCNet trust evaluation
mechanisms with Flower's federated learning server.

Usage:
    python server/run_federated.py --config config/federated.yaml --num_rounds 10

Features:
- Trust-aware client selection and aggregation
- Configurable trust thresholds and evaluation modes
- Production-ready error handling and logging
- Full compatibility with Flower ecosystem
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import flwr as fl

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    # Try to import OmegaConf if available
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = dict
    # We'll need yaml for the fallback path
    import yaml

try:
    from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
    from trust_mcnet.strategies.trust_weighted_strategy import TrustWeightedStrategy
    TRUST_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing TRUST_MCNet modules: {e}")
    print("Please ensure the project is properly installed and paths are correct.")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory before creating FileHandler
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/server.log', mode='a')
        ]
    )


def load_server_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load server configuration with sensible defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'server': {
            'address': '0.0.0.0:8080',
            'num_rounds': 10,
            'min_fit_clients': 2,
            'min_eval_clients': 2,
            'min_available_clients': 2,
            'fraction_fit': 0.8,
            'fraction_eval': 0.2,
            'accept_failures': True
        },
        'trust': {
            'trust_mode': 'hybrid',
            'threshold': 0.5,
            'learning_rate': 0.01,
            'use_dynamic_weights': True,
            'probe_dataset': {
                'name': None,  # Set to a dataset name to enable entropy-based trust
                'batch_size': 32,
                'params': {
                    'train': False,
                    'download': True
                }
            }
        },
        'strategy': {
            'name': 'trust_weighted',
            'trim_ratio': 0.1
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            if OMEGACONF_AVAILABLE:
                # Convert default_config to DictConfig for consistent typing and better validation
                default_cfg = OmegaConf.create(default_config)
                file_cfg = OmegaConf.load(config_path)
                # Merge with defaults (both now properly as DictConfig objects)
                merged_cfg = OmegaConf.merge(default_cfg, file_cfg)
                # Convert back to plain dict with resolved interpolations (resolving any ${var} interpolations)
                config = OmegaConf.to_container(merged_cfg, resolve=True)
            else:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Deep merge for each section instead of shallow merge
                config = default_config.copy()
                for section in ('server', 'trust', 'strategy'):
                    if section in file_config:
                        config[section].update(file_config.get(section, {}))
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            config = default_config
    else:
        config = default_config
        if config_path:
            logger.warning(f"Config file {config_path} not found, using defaults")
    
    return config


def create_trust_evaluator(trust_config: Dict[str, Any], probe_data=None):
    """
    Create and configure trust evaluator.
    
    Args:
        trust_config: Trust configuration section
        probe_data: Optional DataLoader containing a small public dataset for entropy calculation
        
    Returns:
        Configured TrustEvaluator instance
    """
    try:
        trust_eval = TrustEvaluator(
            trust_mode=trust_config.get('trust_mode', 'hybrid'),
            threshold=trust_config.get('threshold', 0.5),
            learning_rate=trust_config.get('learning_rate', 0.01),
            use_dynamic_weights=trust_config.get('use_dynamic_weights', True),
            config=trust_config,  # Pass the entire trust config to evaluator
            probe_data=probe_data  # Pass probe data for improved entropy calculation
        )
        
        has_probe = probe_data is not None
        logger.info(f"Created TrustEvaluator: mode={trust_eval.trust_mode}, "
                   f"threshold={trust_eval.threshold}, probe_data={'available' if has_probe else 'not available'}")
        return trust_eval
        
    except Exception as e:
        logger.error(f"Failed to create TrustEvaluator: {e}")
        raise


def create_strategy(config: Dict[str, Any], trust_evaluator):
    """
    Create trust-weighted strategy for Flower server.
    
    Args:
        config: Server configuration
        trust_evaluator: Trust evaluation instance
        
    Returns:
        Configured TrustWeightedStrategy
    """
    server_config = config['server']
    strategy_config = config.get('strategy', {})
    
    try:
        strategy = TrustWeightedStrategy(
            trust_evaluator=trust_evaluator,
            # Standard FedAvg parameters
            fraction_fit=server_config.get('fraction_fit', 0.8),
            fraction_evaluate=server_config.get('fraction_eval', 0.2),
            min_fit_clients=server_config.get('min_fit_clients', 2),
            min_evaluate_clients=server_config.get('min_eval_clients', 2),
            min_available_clients=server_config.get('min_available_clients', 2),
            accept_failures=server_config.get('accept_failures', True),
            # Strategy-specific parameters
            trim_ratio=strategy_config.get('trim_ratio', 0.1),
        )
        
        logger.info("Created TrustWeightedStrategy with Flower compatibility")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise


def load_probe_data(config):
    """
    Load a small public dataset for entropy calculation.
    
    Args:
        config: Configuration dictionary containing dataset details
        
    Returns:
        DataLoader for the probe dataset, or None if not configured
    """
    if 'probe_dataset' not in config.get('trust', {}):
        return None
    
    probe_config = config['trust'].get('probe_dataset', {})
    dataset_name = probe_config.get('name')
    
    if not dataset_name:
        logger.info("No probe dataset specified in config, entropy will use parameter-histogram fallback")
        return None
    
    try:
        # Import necessary modules for data loading
        from torch.utils.data import DataLoader
        
        # Try to get dataset registry if available
        try:
            from trust_mcnet.datasets import get_dataset
            
            dataset_kwargs = probe_config.get('params', {})
            probe_dataset = get_dataset(dataset_name, **dataset_kwargs)
            
            batch_size = probe_config.get('batch_size', 32)
            probe_loader = DataLoader(
                probe_dataset, 
                batch_size=batch_size,
                shuffle=False
            )
            
            logger.info(f"Loaded probe dataset '{dataset_name}' for entropy calculation "
                       f"({len(probe_dataset)} samples, batch_size={batch_size})")
            
            return probe_loader
            
        except ImportError:
            logger.warning("Dataset registry not available, please implement dataset loading")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to load probe dataset: {e}")
        return None


def main():
    """Main server launcher."""
    parser = argparse.ArgumentParser(description='TRUST_MCNet Flower Server')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds (overrides config)')
    parser.add_argument('--address', type=str, help='Server address (overrides config)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--probe-dataset', type=str, help='Override probe dataset name')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info("="*60)
    logger.info("TRUST_MCNet Flower Federated Learning Server")
    logger.info("="*60)
    
    try:
        # Load configuration
        config = load_server_config(args.config)
        
        # Override with command line arguments
        if args.num_rounds:
            config['server']['num_rounds'] = args.num_rounds
        if args.address:
            config['server']['address'] = args.address
        
        logger.info(f"Server configuration loaded successfully")
        
        # Load probe data if available
        probe_data = load_probe_data(config)
        
        # Override probe dataset from command line if specified
        if args.probe_dataset:
            logger.info(f"Overriding probe dataset with '{args.probe_dataset}' from command line")
            if 'trust' not in config:
                config['trust'] = {}
            if 'probe_dataset' not in config['trust']:
                config['trust']['probe_dataset'] = {}
            config['trust']['probe_dataset']['name'] = args.probe_dataset
            probe_data = load_probe_data(config)
        
        # Create trust evaluator (long-lived object to maintain history)
        trust_evaluator = create_trust_evaluator(config['trust'], probe_data=probe_data)
        
        # Create strategy
        strategy = create_strategy(config, trust_evaluator)
        
        # Configure server
        server_config = fl.server.ServerConfig(
            num_rounds=config['server']['num_rounds']
        )
        
        logger.info(f"Starting Flower server on {config['server']['address']}")
        logger.info(f"Strategy: TrustWeightedStrategy")
        logger.info(f"Trust mode: {config['trust']['trust_mode']}")
        logger.info(f"Rounds: {config['server']['num_rounds']}")
        
        # Log probe dataset info
        probe_dataset_name = config['trust'].get('probe_dataset', {}).get('name')
        if probe_dataset_name and probe_data:
            logger.info(f"Using probe dataset '{probe_dataset_name}' for improved entropy-based trust")
        else:
            logger.info("No probe dataset loaded, using parameter-histogram fallback for entropy")
        
        # Start Flower server
        fl.server.start_server(
            server_address=config['server']['address'],
            config=server_config,
            strategy=strategy,
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
