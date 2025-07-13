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
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/server.log', mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)


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
            'use_dynamic_weights': True
        },
        'strategy': {
            'name': 'trust_weighted',
            'trim_ratio': 0.1
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            if OMEGACONF_AVAILABLE:
                file_config = OmegaConf.load(config_path)
                # Merge with defaults
                config = OmegaConf.merge(default_config, file_config)
                config = OmegaConf.to_container(config, resolve=True)
            else:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Simple merge for fallback
                config = {**default_config, **file_config}
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


def create_trust_evaluator(trust_config: Dict[str, Any]):
    """
    Create and configure trust evaluator.
    
    Args:
        trust_config: Trust configuration section
        
    Returns:
        Configured TrustEvaluator instance
    """
    try:
        trust_eval = TrustEvaluator(
            trust_mode=trust_config.get('trust_mode', 'hybrid'),
            threshold=trust_config.get('threshold', 0.5),
            learning_rate=trust_config.get('learning_rate', 0.01),
            use_dynamic_weights=trust_config.get('use_dynamic_weights', True)
        )
        
        logger.info(f"Created TrustEvaluator: mode={trust_eval.trust_mode}, "
                   f"threshold={trust_eval.threshold}")
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
        )
        
        logger.info("Created TrustWeightedStrategy with Flower compatibility")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise


def main():
    """Main server launcher."""
    parser = argparse.ArgumentParser(description='TRUST_MCNet Flower Server')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds (overrides config)')
    parser.add_argument('--address', type=str, help='Server address (overrides config)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
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
        
        # Create trust evaluator (long-lived object to maintain history)
        trust_evaluator = create_trust_evaluator(config['trust'])
        
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
