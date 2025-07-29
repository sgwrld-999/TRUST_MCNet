#!/usr/bin/env python3
"""
Enhanced API Server Example for TRUST_MCNet

This example demonstrates how to use the enhanced API server with dynamic threshold support.
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    trust_evaluator = TrustEvaluator(config=config)
    storage = TrustStorage(db_path=args.db_path)
    
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
    logger.info("Dynamic threshold enabled: " + 
               str(hasattr(trust_evaluator, '_dynamic_threshold_initialized') and 
                  getattr(trust_evaluator, '_dynamic_threshold_initialized', False)))
    
    # Run server
    api_server.run_server()


if __name__ == "__main__":
    main()
