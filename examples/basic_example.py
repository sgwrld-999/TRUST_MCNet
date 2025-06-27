#!/usr/bin/env python3
"""
Example usage of TRUST MCNet framework.

This script demonstrates how to use the TRUST MCNet framework
for federated learning with anomaly detection.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import TensorDataset

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TRUST_MCNet_Codebase.data.data_loader import ConfigManager
from TRUST_MCNet_Codebase.clients.client import Client
from TRUST_MCNet_Codebase.models.model import MLP, LSTM


def generate_dummy_data(num_samples=1000, num_features=10, anomaly_ratio=0.1):
    """Generate dummy data for demonstration purposes."""
    # Generate normal data
    normal_samples = int(num_samples * (1 - anomaly_ratio))
    normal_data = np.random.normal(0, 1, (normal_samples, num_features))
    normal_labels = np.zeros(normal_samples)
    
    # Generate anomalous data
    anomaly_samples = num_samples - normal_samples
    anomaly_data = np.random.normal(3, 1.5, (anomaly_samples, num_features))
    anomaly_labels = np.ones(anomaly_samples)
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([normal_labels, anomaly_labels])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def split_data_for_clients(X, y, num_clients=3):
    """Split data among multiple clients."""
    num_samples = len(X)
    samples_per_client = num_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:  # Last client gets remaining samples
            end_idx = num_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_X = X[start_idx:end_idx]
        client_y = y[start_idx:end_idx]
        
        # Split into train and test for each client
        train_size = int(0.8 * len(client_X))
        train_X, test_X = client_X[:train_size], client_X[train_size:]
        train_y, test_y = client_y[:train_size], client_y[train_size:]
        
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        client_datasets.append((train_dataset, test_dataset))
    
    return client_datasets


def run_example():
    """Run the example federated learning scenario."""
    print("🚀 Starting TRUST MCNet Example")
    print("=" * 50)
    
    # Load configuration
    config_path = "TRUST_MCNet_Codebase/config/config.yaml"
    config = ConfigManager(config_path)
    print(f"✅ Configuration loaded from: {config_path}")
    
    # Generate dummy data
    print("📊 Generating dummy anomaly detection data...")
    X, y = generate_dummy_data(num_samples=1000, num_features=10)
    print(f"   - Total samples: {len(X)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Normal samples: {(y == 0).sum().item()}")
    print(f"   - Anomalous samples: {(y == 1).sum().item()}")
    
    # Split data for clients
    num_clients = config.get('federated.num_clients', 3)
    client_datasets = split_data_for_clients(X, y, num_clients)
    print(f"📤 Data split among {num_clients} clients")
    
    # Create clients
    clients = []
    for i, (train_dataset, test_dataset) in enumerate(client_datasets):
        # Create model for each client
        model = MLP(
            input_dim=config.get('model.mlp.input_dim', 10),
            hidden_dims=config.get('model.mlp.hidden_dims', [64, 32]),
            output_dim=config.get('model.mlp.output_dim', 2)
        )
        
        # Create client
        client = Client(
            client_id=i + 1,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config.cfg
        )
        clients.append(client)
        print(f"   - Client {i + 1}: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    print("🤖 Clients created successfully")
    
    # Simulate federated learning rounds
    num_rounds = 5
    print(f"\n🔄 Starting {num_rounds} federated learning rounds...")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Train each client
        round_losses = []
        for i, client in enumerate(clients):
            loss = client.train()
            round_losses.append(loss)
            print(f"   Client {i + 1} training loss: {loss:.4f}")
        
        avg_loss = np.mean(round_losses)
        print(f"   Average training loss: {avg_loss:.4f}")
        
        # Evaluate clients (every 2 rounds)
        if (round_num + 1) % 2 == 0:
            print("   📊 Evaluating clients...")
            for i, client in enumerate(clients):
                accuracy = client.test()
                print(f"      Client {i + 1} accuracy: {accuracy:.4f}")
    
    print("\n✨ Federated learning simulation completed!")
    print("=" * 50)
    print("📝 Note: This is a simplified example. A full implementation would include:")
    print("   - Model aggregation (FedAvg, etc.)")
    print("   - Trust mechanism evaluation")
    print("   - Communication protocols")
    print("   - Privacy preservation techniques")
    print("   - Advanced anomaly detection metrics")


if __name__ == "__main__":
    run_example()
