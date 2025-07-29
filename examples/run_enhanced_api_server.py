#!/usr/bin/env python3
"""
Enhanced API Server Example for TRUST_MCNet with Dynamic Threshold

This example demonstrates how to use the enhanced API server with dynamic threshold support properly enabled.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trust_mcnet.api.enhanced_server import EnhancedTrustMCNetAPIServer
from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
from src.trust_mcnet.storage.trust_storage import TrustStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TRUST_MCNet Enhanced API Server")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/trust.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to listen on"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="trust_mcnet.db",
        help="Path to SQLite database file"
    )
    
    parser.add_argument(
        "--enable-dynamic-threshold",
        action="store_true",
        help="Enable dynamic threshold support"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    trust_evaluator = TrustEvaluator(config=config)
    storage = TrustStorage(db_path=args.db_path)
    
    # Explicitly initialize dynamic threshold if requested
    if args.enable_dynamic_threshold:
        try:
            # Access the private method using Python's name mangling
            method = getattr(trust_evaluator, f"_{TrustEvaluator.__name__}__init_dynamic_threshold_system")
            method()
            logger.info("Dynamic threshold system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dynamic threshold: {e}")
    
    # Check if dynamic threshold is actually enabled
    dynamic_enabled = hasattr(trust_evaluator, '_dynamic_threshold_initialized') and \
                      getattr(trust_evaluator, '_dynamic_threshold_initialized', False)
    
    # Create API server
    api_server = EnhancedTrustMCNetAPIServer(
        trust_evaluator=trust_evaluator,
        storage=storage,
        host=args.host,
        port=args.port
    )
    
    logger.info(f"Starting Enhanced API Server on {args.host}:{args.port}")
    logger.info(f"Trust evaluator mode: {trust_evaluator.trust_mode}")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Dynamic threshold enabled: {dynamic_enabled}")
    
    # Run server
    api_server.run_server()


if __name__ == "__main__":
    main()
