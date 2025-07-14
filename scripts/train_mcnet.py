#!/usr/bin/env python3
"""
TRUST_MCNet Training Orchestration Script

Single CLI entry-point for running federated learning experiments
with trust-based client selection and aggregation.

Usage:
    python scripts/train_mcnet.py --dataset ton_iot --rounds 10
    MCNET_DATA=/path/to/data python scripts/train_mcnet.py --dataset edge_iiot --rounds 5
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import ray
    import flwr as fl
    import torch
    import numpy as np
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Import TRUST_MCNet components
try:
    from datasets import get as get_dataset, get_data_root, list_datasets
    from src.trust_mcnet.strategies.unified_trust_strategy import UnifiedTrustStrategy
    from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
    from src.trust_mcnet.models.simple_mlp import SimpleMLP
except ImportError as e:
    print(f"Failed to import TRUST_MCNet components: {e}")
    print("Ensure the project structure is correct and all modules are available.")
    sys.exit(1)


class MCNetRayClient(fl.client.NumPyClient):
    """Ray-compatible Flower client for TRUST_MCNet simulation."""
    
    def __init__(self, cid: str, dataset, model_config: Dict[str, Any]):
        """
        Initialize federated client.
        
        Args:
            cid: Client ID
            dataset: Dataset instance
            model_config: Model configuration
        """
        self.cid = cid
        self.dataset = dataset
        
        # Create model
        self.model = SimpleMLP(
            input_dim=dataset.input_dim,
            hidden_dims=model_config.get("hidden_dims", [64, 32]),
            num_classes=dataset.num_classes,
            dropout_rate=model_config.get("dropout_rate", 0.1)
        )
        
        # Training configuration
        self.epochs = model_config.get("local_epochs", 1)
        self.lr = model_config.get("learning_rate", 0.001)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Get data loaders
        self.train_loader = dataset.train_loader()
        self.test_loader = dataset.test_loader()
        
        logging.info(f"Initialized client {cid} with {len(dataset)} samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> tuple:
        """Train the model locally."""
        self.set_parameters(parameters)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Return updated parameters and metrics
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            {"train_loss": avg_loss, "client_id": self.cid}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> tuple:
        """Evaluate the model locally."""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, total, {"accuracy": accuracy, "client_id": self.cid}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)
        else:
            logging.warning(f"Config file {config_path} not found. Using defaults.")
            return get_default_config()
    except Exception as e:
        logging.error(f"Error loading config: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "trust": {
            "mode": "hybrid",
            "threshold": 0.5,
            "gamma_shap": 0.25,
            "lr": {
                "enable": True,
                "base": 0.001,
                "beta": 0.5,
                "mu": 0.5
            },
            "aggregation": {
                "trim_ratio": 0.2
            }
        },
        "federated": {
            "num_clients": 5,
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.2,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2
        },
        "model": {
            "hidden_dims": [64, 32],
            "dropout_rate": 0.1,
            "local_epochs": 1,
            "learning_rate": 0.001
        },
        "batch_size": 32
    }


def create_client_fn(dataset, model_config: Dict[str, Any]):
    """Create client function for Flower simulation."""
    def client_fn(cid: str) -> MCNetRayClient:
        return MCNetRayClient(cid, dataset, model_config)
    return client_fn


def run_simulation(dataset_name: str, rounds: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run federated learning simulation.
    
    Args:
        dataset_name: Name of dataset to use
        rounds: Number of federated rounds
        config: Configuration dictionary
        
    Returns:
        Simulation results
    """
    logging.info(f"Starting TRUST_MCNet simulation: {dataset_name}, {rounds} rounds")
    
    # Initialize dataset
    data_root = get_data_root()
    logging.info(f"Using data root: {data_root}")
    
    dataset = get_dataset(
        dataset_name,
        batch_size=config.get("batch_size", 32),
        data_root=data_root
    )
    
    logging.info(f"Dataset info: {dataset.get_info()}")
    
    # Create trust evaluator
    trust_config = config.get("trust", {})
    trust_eval = TrustEvaluator(
        trust_mode=trust_config.get("mode", "hybrid"),
        threshold=trust_config.get("threshold", 0.5),
        learning_rate=trust_config.get("lr", {}).get("base", 0.001),
        use_dynamic_weights=True,
        config=trust_config
    )
    
    logging.info(f"Trust evaluator: mode={trust_eval.trust_mode}, threshold={trust_eval.threshold}")
    
    # Create strategy
    fed_config = config.get("federated", {})
    strategy = UnifiedTrustStrategy(
        trust_evaluator=trust_eval,
        fraction_fit=fed_config.get("fraction_fit", 0.8),
        fraction_evaluate=fed_config.get("fraction_evaluate", 0.2),
        min_fit_clients=fed_config.get("min_fit_clients", 2),
        min_evaluate_clients=fed_config.get("min_evaluate_clients", 2),
        min_available_clients=fed_config.get("min_available_clients", 2)
    )
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Create client function
    client_fn = create_client_fn(dataset, config.get("model", {}))
    
    # Start simulation
    start_time = time.time()
    
    try:
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=fed_config.get("num_clients", 5),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            ray_init_args={"ignore_reinit_error": True}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract results
        results = {
            "dataset": dataset_name,
            "rounds": rounds,
            "duration": duration,
            "history": hist,
            "final_metrics": extract_final_metrics(hist),
            "trust_summary": get_trust_summary(strategy, trust_eval)
        }
        
        logging.info(f"Simulation completed in {duration:.2f} seconds")
        return results
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise
    finally:
        # Cleanup Ray if we initialized it
        try:
            ray.shutdown()
        except:
            pass


def extract_final_metrics(hist) -> Dict[str, Any]:
    """Extract final metrics from Flower history."""
    metrics = {}
    
    if hasattr(hist, 'metrics_distributed') and hist.metrics_distributed:
        latest_metrics = hist.metrics_distributed.get('accuracy', [])
        if latest_metrics:
            metrics['final_accuracy'] = latest_metrics[-1][1]
    
    if hasattr(hist, 'losses_distributed') and hist.losses_distributed:
        latest_loss = hist.losses_distributed[-1][1] if hist.losses_distributed else None
        if latest_loss:
            metrics['final_loss'] = latest_loss
    
    return metrics


def get_trust_summary(strategy, trust_eval) -> Dict[str, Any]:
    """Get trust evaluation summary."""
    try:
        if hasattr(trust_eval, 'get_trust_statistics'):
            return trust_eval.get_trust_statistics({})
        return {"status": "Trust statistics not available"}
    except Exception as e:
        return {"error": str(e)}


def save_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save simulation results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Make results JSON serializable
    serializable_results = {}
    for key, value in results.items():
        if key == "history":
            # Skip complex Flower history object
            continue
        try:
            json.dumps(value)  # Test if serializable
            serializable_results[key] = value
        except (TypeError, ValueError):
            serializable_results[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return filepath


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TRUST_MCNet Federated Learning Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/train_mcnet.py --dataset ton_iot --rounds 5
  MCNET_DATA=/path/to/data python scripts/train_mcnet.py --dataset edge_iiot --rounds 10
  python scripts/train_mcnet.py --dataset medbiot --rounds 3 --config config/custom.yaml

Available datasets: {', '.join(list_datasets()) if list_datasets() else 'Loading...'}
        """
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ton_iot", "edge_iiot", "medbiot"],
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)"
    )
    parser.add_argument(
        "--config",
        default="config/trust.yaml",
        help="Configuration file path (default: config/trust.yaml)"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Run simulation
        results = run_simulation(args.dataset, args.rounds, config)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TRUST_MCNet Simulation Complete")
        print(f"{'='*60}")
        print(f"Dataset: {results['dataset']}")
        print(f"Rounds: {results['rounds']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            if 'final_accuracy' in metrics:
                print(f"Final Accuracy: {metrics['final_accuracy']:.4f}")
            if 'final_loss' in metrics:
                print(f"Final Loss: {metrics['final_loss']:.4f}")
        
        print(f"Trust Summary: {results.get('trust_summary', 'N/A')}")
        
        # Save results if requested
        if args.save_results:
            filepath = save_results(results, args.output)
            print(f"Results saved to: {filepath}")
        
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
