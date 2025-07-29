#!/usr/bin/env python3
"""
Script to run the TRUST_MCNet simulation with dynamic threshold enabled
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import logging
from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator

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

def main():
    # Load configuration
    config_path = "config/trust.yaml"
    config = load_config(config_path)
    
    # Check if dynamic threshold is enabled in config
    dynamic_threshold_enabled = config.get('trust', {}).get('dynamic_threshold', {}).get('enabled', False)
    print(f"Dynamic threshold enabled in config: {dynamic_threshold_enabled}")
    
    # Initialize trust evaluator
    trust_evaluator = TrustEvaluator(config=config)
    
    # Print the config and trust.dynamic_threshold structure
    print("\nConfig structure:")
    print(f"trust.dynamic_threshold: {config.get('trust', {}).get('dynamic_threshold', {})}")
    print(f"Trust evaluator config: {trust_evaluator.config}")
    
    # Manually initialize dynamic threshold system
    print("\nAttempting to initialize dynamic threshold system...")
    try:
        # Need to access the private method directly
        method = getattr(trust_evaluator, f"_{TrustEvaluator.__name__}__init_dynamic_threshold_system")
        method()
        print("Successfully called __init_dynamic_threshold_system")
    except Exception as e:
        print(f"Error initializing dynamic threshold system: {e}")
    
    # Check if dynamic threshold is enabled in trust evaluator
    dynamic_threshold_initialized = hasattr(trust_evaluator, '_dynamic_threshold_initialized') and \
                                   getattr(trust_evaluator, '_dynamic_threshold_initialized', False)
    print(f"Dynamic threshold initialized in trust evaluator: {dynamic_threshold_initialized}")
    
    # Start simulation if dynamic threshold is enabled
    if dynamic_threshold_initialized:
        print("Starting simulation with dynamic threshold enabled...")
        # Import and run simulation here
        from examples.start_simulation import main as run_simulation
        run_simulation()
    else:
        print("Dynamic threshold not properly initialized")

if __name__ == "__main__":
    main()
