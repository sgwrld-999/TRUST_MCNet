"""
Flwr simulation runner for TRUST-MCNet federated learning.

This script sets up and runs federated learning simulation using Flwr
with IoT device considerations and trust mechanisms.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, Subset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import flwr as fl
    from flwr.simulation import start_simulation
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    print("Flwr not available. Please install with: pip install flwr")

from TRUST_MCNet_Codebase.models.model import MLP, LSTM
from TRUST_MCNet_Codebase.data.data_loader import ConfigManager
from TRUST_MCNet_Codebase.server.flwr_server import create_flwr_server
from TRUST_MCNet_Codebase.clients.flwr_client import create_flwr_client


class SyntheticIoTDataset(Dataset):
    """
    Synthetic dataset for IoT anomaly detection simulation.
    
    Generates synthetic sensor data with normal and anomalous patterns.
    """
    
    def __init__(self, num_samples: int = 1000, input_dim: int = 10, anomaly_ratio: float = 0.1):
        """
        Initialize synthetic IoT dataset.
        
        Args:
            num_samples: Number of samples to generate
            input_dim: Dimensionality of input features
            anomaly_ratio: Ratio of anomalous samples
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.anomaly_ratio = anomaly_ratio
        
        # Generate data
        self.data, self.labels = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic IoT sensor data."""
        np.random.seed(42)
        
        num_anomalies = int(self.num_samples * self.anomaly_ratio)
        num_normal = self.num_samples - num_anomalies
        
        # Generate normal samples (Gaussian distribution)
        normal_data = np.random.normal(0.5, 0.15, (num_normal, self.input_dim))
        normal_labels = np.zeros(num_normal)
        
        # Generate anomalous samples (different distribution)
        anomaly_data = np.random.normal(0.8, 0.3, (num_anomalies, self.input_dim))
        # Add some extreme outliers
        anomaly_data[::10] = np.random.uniform(-2, 2, (len(anomaly_data[::10]), self.input_dim))
        anomaly_labels = np.ones(num_anomalies)
        
        # Combine and shuffle
        all_data = np.vstack([normal_data, anomaly_data])
        all_labels = np.hstack([normal_labels, anomaly_labels])
        
        # Shuffle
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]
        
        return torch.FloatTensor(all_data), torch.LongTensor(all_labels)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def create_client_datasets(
    full_dataset: Dataset, 
    num_clients: int, 
    data_distribution: str = "iid"
) -> Tuple[list, list]:
    """
    Split dataset among clients with different data distributions.
    
    Args:
        full_dataset: Complete dataset to split
        num_clients: Number of clients
        data_distribution: "iid" for IID or "non_iid" for non-IID distribution
        
    Returns:
        Tuple of (train_datasets, test_datasets) for each client
    """
    total_size = len(full_dataset)
    
    if data_distribution == "iid":
        # IID distribution: equal random splits
        client_sizes = [total_size // num_clients] * num_clients
        # Handle remainder
        for i in range(total_size % num_clients):
            client_sizes[i] += 1
        
        # Random split
        client_datasets = random_split(full_dataset, client_sizes)
        
    elif data_distribution == "non_iid":
        # Non-IID distribution: clients have different class distributions
        # Sort by labels for non-IID split
        indices = list(range(total_size))
        labels = [full_dataset[i][1].item() for i in indices]
        sorted_indices = sorted(indices, key=lambda x: labels[x])
        
        # Assign consecutive samples to clients (creates label skew)
        client_datasets = []
        samples_per_client = total_size // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            if i == num_clients - 1:  # Last client gets remaining samples
                end_idx = total_size
            
            client_indices = sorted_indices[start_idx:end_idx]
            client_datasets.append(Subset(full_dataset, client_indices))
    
    else:
        raise ValueError(f"Unknown data distribution: {data_distribution}")
    
    # Split each client's data into train/test
    train_datasets = []
    test_datasets = []
    
    for client_dataset in client_datasets:
        client_size = len(client_dataset)
        train_size = int(0.8 * client_size)
        test_size = client_size - train_size
        
        client_train, client_test = random_split(client_dataset, [train_size, test_size])
        train_datasets.append(client_train)
        test_datasets.append(client_test)
    
    return train_datasets, test_datasets


def client_fn(cid: str, config: Dict[str, Any]) -> fl.client.Client:
    """
    Create client function for Flwr simulation.
    
    Args:
        cid: Client ID
        config: Configuration dictionary
        
    Returns:
        Flwr client instance
    """
    client_id = int(cid)
    
    # Get datasets for this client
    train_dataset = config["train_datasets"][client_id]
    test_dataset = config["test_datasets"][client_id]
    
    # Create model copy for this client
    model_config = config["model_config"]
    if model_config["type"] == "MLP":
        model = MLP(
            input_dim=model_config["mlp"]["input_dim"],
            output_dim=model_config["mlp"]["output_dim"]
        )
    elif model_config["type"] == "LSTM":
        model = LSTM(
            input_dim=model_config["lstm"]["input_dim"],
            hidden_dim=model_config["lstm"]["hidden_dim"],
            num_layers=model_config["lstm"]["num_layers"],
            output_dim=model_config["lstm"]["output_dim"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Create Flwr client
    return create_flwr_client(
        client_id=cid,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config["client_config"]
    )


def run_flwr_simulation(config_path: str):
    """
    Run Flwr federated learning simulation.
    
    Args:
        config_path: Path to configuration file
    """
    if not FLWR_AVAILABLE:
        raise ImportError("Flwr is not available. Please install with: pip install flwr")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting TRUST-MCNet Flwr simulation")
    logger.info(f"Configuration: {config_path}")
    
    # Create synthetic IoT dataset
    dataset_size = config.get('data', {}).get('dataset_size', 5000)
    input_dim = config.get('model', {}).get('mlp', {}).get('input_dim', 10)
    anomaly_ratio = config.get('data', {}).get('anomaly_ratio', 0.1)
    
    full_dataset = SyntheticIoTDataset(
        num_samples=dataset_size,
        input_dim=input_dim,
        anomaly_ratio=anomaly_ratio
    )
    
    logger.info(f"Created dataset with {len(full_dataset)} samples")
    
    # Split data among clients
    num_clients = config.get('federated', {}).get('num_clients', 10)
    data_distribution = config.get('data', {}).get('distribution', 'iid')
    
    train_datasets, test_datasets = create_client_datasets(
        full_dataset, num_clients, data_distribution
    )
    
    logger.info(f"Split data among {num_clients} clients ({data_distribution} distribution)")
    
    # Create global model for server
    model_config = config.get('model', {})
    if model_config.get('type') == 'MLP':
        global_model = MLP(
            input_dim=model_config.get('mlp', {}).get('input_dim', 10),
            output_dim=model_config.get('mlp', {}).get('output_dim', 2)
        )
    elif model_config.get('type') == 'LSTM':
        global_model = LSTM(
            input_dim=model_config.get('lstm', {}).get('input_dim', 10),
            hidden_dim=model_config.get('lstm', {}).get('hidden_dim', 64),
            num_layers=model_config.get('lstm', {}).get('num_layers', 2),
            output_dim=model_config.get('lstm', {}).get('output_dim', 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.get('type')}")
    
    logger.info(f"Created global model: {model_config.get('type')}")
    
    # Create server
    server, server_config = create_flwr_server(global_model, config)
    
    # Configuration for client function
    client_config = {
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "model_config": model_config,
        "client_config": config
    }
    
    # Create client function with configuration
    def configured_client_fn(cid: str) -> fl.client.Client:
        return client_fn(cid, client_config)
    
    # Simulation configuration
    num_rounds = config.get('federated', {}).get('num_rounds', 100)
    
    logger.info(f"Starting simulation with {num_clients} clients for {num_rounds} rounds")
    
    # Start simulation
    try:
        history = start_simulation(
            client_fn=configured_client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=server.strategy,
            client_resources={
                "num_cpus": 1.0,
                "num_gpus": 0.0  # No GPU for IoT simulation
            },
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "log_to_driver": False
            }
        )
        
        logger.info("Simulation completed successfully")
        
        # Log final results
        if history.metrics_centralized:
            final_accuracy = history.metrics_centralized.get('accuracy', [])
            if final_accuracy:
                logger.info(f"Final accuracy: {final_accuracy[-1][1]:.4f}")
        
        # Save results
        results_path = "simulation_results.txt"
        with open(results_path, 'w') as f:
            f.write(f"TRUST-MCNet Flwr Simulation Results\n")
            f.write(f"=====================================\n")
            f.write(f"Clients: {num_clients}\n")
            f.write(f"Rounds: {num_rounds}\n")
            f.write(f"Strategy: {config.get('federated', {}).get('strategy', 'FedAdam')}\n")
            f.write(f"Data distribution: {data_distribution}\n")
            f.write(f"Model type: {model_config.get('type')}\n")
            
            if history.metrics_centralized:
                f.write(f"\nCentralized Metrics:\n")
                for metric_name, values in history.metrics_centralized.items():
                    if values:
                        f.write(f"{metric_name}: {values[-1][1]:.4f}\n")
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def main():
    """Main entry point for Flwr simulation."""
    parser = argparse.ArgumentParser(
        description="TRUST-MCNet Flwr Federated Learning Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="TRUST_MCNet_Codebase/config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--clients",
        type=int,
        help="Number of clients (overrides config)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Number of rounds (overrides config)"
    )
    parser.add_argument(
        "--data-distribution",
        type=str,
        choices=["iid", "non_iid"],
        help="Data distribution type (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.clients or args.rounds or args.data_distribution:
        # Load and modify config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.clients:
            config.setdefault('federated', {})['num_clients'] = args.clients
        
        if args.rounds:
            config.setdefault('federated', {})['num_rounds'] = args.rounds
        
        if args.data_distribution:
            config.setdefault('data', {})['distribution'] = args.data_distribution
        
        # Save modified config
        modified_config_path = "modified_config.yaml"
        with open(modified_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_path = modified_config_path
    else:
        config_path = args.config
    
    try:
        run_flwr_simulation(config_path)
    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
