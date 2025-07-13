#!/usr/bin/env python3
"""
TRUST-MCNet Main Entry Point
============================

This script provides the main entry point for running TRUST-MCNet simulations
and Flower federated learning server.

Usage:
    # Run standard simulation
    python main.py [simulation_options]
    
    # Run Flower trust-weighted server
    python main.py --mode flower_server --num_rounds 10 --verbose
    
    # Run test client for Flower server
    python main.py --mode test_client --client_id 0
    
For detailed options, see examples/start_simulation.py
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import sys
import os
import argparse
import logging
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional

# Add src and examples to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
examples_path = current_dir / "examples"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(examples_path))

# Conditional imports for Flower functionality
try:
    import flwr as fl
    import yaml
    import numpy as np
    from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
    
    # Import unified trust strategy with backward compatibility fallback
    try:
        from trust_mcnet.strategies.unified_trust_strategy import (
            UnifiedTrustStrategy, 
            TrustWeightedStrategy,  # Backward compatibility alias
            AdaptiveTrustStrategy   # Backward compatibility alias
        )
        from trust_mcnet.monitoring.trust_dashboard import TrustDashboard
    except ImportError:
        # Fallback to legacy strategies
        try:
            from trust_mcnet.strategies.trust_weighted_strategy import TrustWeightedStrategy
            from trust_mcnet.strategies.adaptive_trust_strategy import AdaptiveTrustStrategy
            from trust_mcnet.monitoring.trust_dashboard import TrustDashboard
            UnifiedTrustStrategy = None  # Mark as unavailable
        except ImportError:
            # Last resort: direct import
            import importlib.util
            
            # Import TrustWeightedStrategy
            spec = importlib.util.spec_from_file_location(
                'trust_weighted_strategy', 
                str(src_path / 'trust_mcnet/strategies/trust_weighted_strategy.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TrustWeightedStrategy = module.TrustWeightedStrategy
            
            # Import AdaptiveTrustStrategy
            try:
                spec = importlib.util.spec_from_file_location(
                    'adaptive_trust_strategy', 
                    str(src_path / 'trust_mcnet/strategies/adaptive_trust_strategy.py')
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AdaptiveTrustStrategy = module.AdaptiveTrustStrategy
            except Exception:
                AdaptiveTrustStrategy = TrustWeightedStrategy  # Fallback
            
            UnifiedTrustStrategy = None  # Mark as unavailable for legacy fallback
        
        # Import TrustDashboard
        try:
            spec = importlib.util.spec_from_file_location(
                'trust_dashboard', 
                str(src_path / 'trust_mcnet/monitoring/trust_dashboard.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TrustDashboard = module.TrustDashboard
        except Exception:
            TrustDashboard = None  # Monitoring not available
    
    FLOWER_AVAILABLE = True
except ImportError as e:
    FLOWER_AVAILABLE = False


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


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def run_flower_server(args) -> int:
    """Run the trust-weighted Flower server."""
    if not FLOWER_AVAILABLE:
        print("Error: Flower and required dependencies not available")
        print("Please install with: pip install flwr torch numpy")
        return 1
    
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.num_rounds:
        config.setdefault('federated', {})['num_rounds'] = args.num_rounds
    if args.address:
        config.setdefault('federated', {}).setdefault('server', {})['address'] = args.address
    
    try:
        # Create trust evaluator
        trust_config = config.get('trust', {})
        trust_eval = TrustEvaluator(
            trust_mode=trust_config.get('trust_mode', 'hybrid'),
            threshold=trust_config.get('threshold', 0.5),
            learning_rate=trust_config.get('learning_rate', 0.01),
            use_dynamic_weights=trust_config.get('use_dynamic_weights', True)
        )
        
        logger.info(f"Created TrustEvaluator: mode={trust_eval.trust_mode}, "
                   f"threshold={trust_eval.threshold}")
        
        # Create strategy - prefer unified strategy
        fed_config = config.get('federated', {})
        use_adaptive = fed_config.get('use_adaptive_strategy', False)
        
        if UnifiedTrustStrategy is not None:
            # Use unified strategy (recommended)
            strategy = UnifiedTrustStrategy(
                trust_evaluator=trust_eval,
                enable_adaptation=use_adaptive,
                target_accuracy=fed_config.get('target_accuracy', 0.85),
                threshold_adaptation_rate=fed_config.get('adaptation_rate', 0.05),
                max_threshold=fed_config.get('max_threshold', 0.9),
                min_threshold=fed_config.get('min_threshold', 0.3),
                performance_window=fed_config.get('performance_window', 5),
                adaptation_patience=fed_config.get('adaptation_patience', 3),
                fraction_fit=fed_config.get('fraction_fit', 0.8),
                fraction_evaluate=fed_config.get('fraction_evaluate', 0.2),
                min_fit_clients=fed_config.get('min_fit_clients', 2),
                min_evaluate_clients=fed_config.get('min_evaluate_clients', 2),
                min_available_clients=fed_config.get('min_available_clients', 2),
                accept_failures=fed_config.get('server', {}).get('accept_failures', True),
            )
            strategy_mode = "Adaptive" if use_adaptive else "Standard"
            logger.info(f"Created UnifiedTrustStrategy ({strategy_mode} mode)")
            
        elif use_adaptive and 'AdaptiveTrustStrategy' in globals():
            # Fallback to legacy adaptive strategy
            strategy = AdaptiveTrustStrategy(
                trust_evaluator=trust_eval,
                target_accuracy=fed_config.get('target_accuracy', 0.85),
                threshold_adaptation_rate=fed_config.get('adaptation_rate', 0.05),
                max_threshold=fed_config.get('max_threshold', 0.9),
                min_threshold=fed_config.get('min_threshold', 0.3),
                performance_window=fed_config.get('performance_window', 5),
                adaptation_patience=fed_config.get('adaptation_patience', 3),
                fraction_fit=fed_config.get('fraction_fit', 0.8),
                fraction_evaluate=fed_config.get('fraction_evaluate', 0.2),
                min_fit_clients=fed_config.get('min_fit_clients', 2),
                min_evaluate_clients=fed_config.get('min_evaluate_clients', 2),
                min_available_clients=fed_config.get('min_available_clients', 2),
                accept_failures=fed_config.get('server', {}).get('accept_failures', True),
            )
            logger.info("Created legacy AdaptiveTrustStrategy")
        else:
            # Fallback to standard trust-weighted strategy
            strategy = TrustWeightedStrategy(
                trust_evaluator=trust_eval,
                fraction_fit=fed_config.get('fraction_fit', 0.8),
                fraction_evaluate=fed_config.get('fraction_evaluate', 0.2),
                min_fit_clients=fed_config.get('min_fit_clients', 2),
                min_evaluate_clients=fed_config.get('min_evaluate_clients', 2),
                min_available_clients=fed_config.get('min_available_clients', 2),
                accept_failures=fed_config.get('server', {}).get('accept_failures', True),
            )
            logger.info("Created legacy TrustWeightedStrategy")
        
        # Setup monitoring dashboard if available
        dashboard = None
        if TrustDashboard is not None and fed_config.get('enable_monitoring', False):
            try:
                dashboard = TrustDashboard(
                    strategy=strategy,
                    output_dir=fed_config.get('dashboard_output_dir', 'trust_dashboard')
                )
                dashboard.start_monitoring(
                    update_interval=fed_config.get('dashboard_update_interval', 30)
                )
                logger.info("Started trust monitoring dashboard")
            except Exception as e:
                logger.warning(f"Failed to start monitoring dashboard: {e}")
                dashboard = None
        
        # Configure server
        num_rounds = fed_config.get('num_rounds', 5)
        server_address = fed_config.get('server', {}).get('address', '0.0.0.0:8080')
        
        server_config = fl.server.ServerConfig(num_rounds=num_rounds)
        
        logger.info(f"Starting Flower server on {server_address}")
        logger.info(f"Trust mode: {trust_config.get('trust_mode', 'hybrid')}")
        logger.info(f"Rounds: {num_rounds}")
        
        # Start Flower server
        try:
            fl.server.start_server(
                server_address=server_address,
                config=server_config,
                strategy=strategy,
            )
        finally:
            # Cleanup dashboard
            if dashboard is not None:
                try:
                    dashboard.stop_monitoring()
                    logger.info("Stopped trust monitoring dashboard")
                except Exception as e:
                    logger.warning(f"Error stopping dashboard: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Server failed: {e}")
        return 1


class SimpleTestClient(fl.client.NumPyClient):
    """Simple test client for trust-weighted server validation."""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.model_params = self._initialize_params()
        
    def _initialize_params(self) -> list:
        """Initialize simple model parameters."""
        # Simple 2-layer MLP for MNIST-like data (784 -> 128 -> 10)
        np.random.seed(int(self.client_id))
        
        params = [
            np.random.normal(0, 0.1, (784, 128)).astype(np.float32),  # W1
            np.zeros(128, dtype=np.float32),                          # b1
            np.random.normal(0, 0.1, (128, 10)).astype(np.float32),  # W2
            np.zeros(10, dtype=np.float32)                            # b2
        ]
        
        return params
    
    def get_parameters(self, config: Dict[str, Any]) -> list:
        """Return current model parameters."""
        return self.model_params.copy()
    
    def fit(self, parameters: list, config: Dict[str, Any]):
        """Simulate local training."""
        # Update local parameters
        self.model_params = [p.copy() for p in parameters]
        
        # Simulate training by adding small noise
        noise_scale = 0.01
        for i, param in enumerate(self.model_params):
            noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
            self.model_params[i] += noise
        
        # Simulate training metrics
        num_samples = np.random.randint(50, 200)
        accuracy = np.random.uniform(0.7, 0.95)
        loss = np.random.uniform(0.1, 0.5)
        
        # Add client-specific bias for trust evaluation testing
        client_bias = int(self.client_id) * self.config.get('simulation', {}).get('client_simulation', {}).get('client_bias_factor', 0.05)
        accuracy = max(0.0, min(1.0, accuracy - client_bias))
        loss = max(0.0, loss + client_bias)
        
        metrics = {
            'accuracy': float(accuracy),
            'train_loss': float(loss),
            'client_id': self.client_id,
            'epochs_completed': 3
        }
        
        print(f"Client {self.client_id}: Training completed. Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")
        
        return self.model_params, num_samples, metrics
    
    def evaluate(self, parameters: list, config: Dict[str, Any]):
        """Simulate evaluation."""
        # Update parameters for evaluation
        self.model_params = [p.copy() for p in parameters]
        
        # Simulate evaluation metrics
        num_samples = np.random.randint(20, 50)
        accuracy = np.random.uniform(0.6, 0.9)
        loss = np.random.uniform(0.2, 0.6)
        
        metrics = {
            'accuracy': float(accuracy),
            'client_id': self.client_id
        }
        
        print(f"Client {self.client_id}: Evaluation completed. Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")
        
        return float(loss), num_samples, metrics


def run_test_client(args) -> int:
    """Run a test client for the Flower server."""
    if not FLOWER_AVAILABLE:
        print("Error: Flower and required dependencies not available")
        return 1
    
    config = load_config()
    client_config = config.get('simulation', {}).get('client_simulation', {})
    server_address = client_config.get('server_address', 'localhost:8080')
    
    print(f"Starting test client {args.client_id}")
    print(f"Connecting to server: {server_address}")
    
    try:
        # Create client
        client = SimpleTestClient(args.client_id, config)
        
        # Start client
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        return 0
        
    except Exception as e:
        print(f"Client failed: {e}")
        return 1

def main():
    """Main entry point for TRUST_MCNet with multiple operational modes."""
    # Check if help is requested for simulation mode
    if len(sys.argv) >= 2 and sys.argv[1] == 'simulation' and '--help' in sys.argv:
        # Pass through to start_simulation.py for help
        os.chdir(current_dir)
        cmd = [sys.executable, str(examples_path / "start_simulation.py"), '--help']
        subprocess.run(cmd)
        return 0
    
    parser = argparse.ArgumentParser(
        description="TRUST_MCNet - Trust-aware Multi-Criteria Network for Federated Learning"
    )
    parser.add_argument(
        'mode',
        choices=['simulation', 'flower_server', 'test_client'],
        help='Operation mode'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Flower server specific arguments
    parser.add_argument(
        '--num-rounds',
        type=int,
        help='Number of federated learning rounds'
    )
    parser.add_argument(
        '--address',
        type=str,
        help='Server address (e.g., 0.0.0.0:8080)'
    )
    
    # Test client specific arguments
    parser.add_argument(
        '--client-id',
        type=str,
        default='1',
        help='Client identifier for test client mode'
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.mode == 'simulation':
            # Change to project root directory
            os.chdir(current_dir)
            
            # Run the simulation script with remaining arguments
            cmd = [sys.executable, str(examples_path / "start_simulation.py")] + unknown_args
            
            # Execute the simulation
            result = subprocess.run(cmd, check=True)
            return result.returncode
        
        elif args.mode == 'flower_server':
            return run_flower_server(args)
        
        elif args.mode == 'test_client':
            return run_test_client(args)
        
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"Error: Simulation failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logging.error(f"Failed to run {args.mode}: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
