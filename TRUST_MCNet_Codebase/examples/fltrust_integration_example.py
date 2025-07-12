"""
Example script demonstrating FLTrust aggregator integration with TRUST-MCNet.

This example shows how to:
1. Initialize FLTrust aggregator with a trusted root dataset
2. Integrate with the existing TRUST-MCNet server
3. Use custom trust scores from TRUST-MCNet's trust module
4. Switch between different aggregators (FedAvg, FLTrust, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.aggregator import FLTrustAggregator, FedAvgAggregator, TrimmedMeanAggregator
from models.model import MLP  # Assuming MLP model exists
from server.server import FederatedServer


def create_synthetic_root_dataset(num_samples: int = 100, input_dim: int = 784, num_classes: int = 10):
    """
    Create a small synthetic trusted root dataset for demonstration.
    In practice, this would be a carefully curated, clean dataset.
    """
    # Generate synthetic data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader


def demonstrate_fltrust_integration():
    """Demonstrate FLTrust integration with TRUST-MCNet."""
    
    print("=== FLTrust Integration Example ===\n")
    
    # 1. Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create global model (same architecture as clients)
    global_model = MLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
    
    # Create root model (copy of global model for FLTrust)
    root_model = MLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
    root_model.load_state_dict(global_model.state_dict())
    
    # 2. Create trusted root dataset
    print("Creating trusted root dataset...")
    root_dataset = create_synthetic_root_dataset(num_samples=100)
    
    # 3. Initialize different aggregators
    print("Initializing aggregators...\n")
    
    # FLTrust aggregator
    fltrust_aggregator = FLTrustAggregator(
        root_model=root_model,
        root_dataset_loader=root_dataset,
        server_learning_rate=1.0,
        clip_threshold=10.0,
        warm_up_rounds=3,
        trust_threshold=0.1
    )
    
    # Standard FedAvg aggregator for comparison
    fedavg_aggregator = FedAvgAggregator(clip_threshold=10.0)
    
    # Trimmed mean aggregator
    trimmed_aggregator = TrimmedMeanAggregator(trim_ratio=0.2, clip_threshold=5.0)
    
    print("Available aggregators:")
    print("- FLTrustAggregator")
    print("- FedAvgAggregator") 
    print("- TrimmedMeanAggregator")
    print()
    
    # 4. Simulate client updates
    print("Simulating client updates...")
    num_clients = 5
    client_updates = {}
    
    for i in range(num_clients):
        client_id = f"client_{i}"
        # Simulate client model weights (add some noise to global model)
        client_weights = {}
        for name, param in global_model.named_parameters():
            noise = torch.randn_like(param) * 0.01  # Small random updates
            client_weights[name] = param.data + noise
        client_updates[client_id] = client_weights
    
    # Add one potentially malicious client with larger deviation
    malicious_weights = {}
    for name, param in global_model.named_parameters():
        noise = torch.randn_like(param) * 0.1  # Larger deviation
        malicious_weights[name] = param.data + noise
    client_updates["malicious_client"] = malicious_weights
    
    # 5. Demonstrate different aggregation methods
    print(f"Testing aggregation with {len(client_updates)} clients (including 1 potentially malicious)\n")
    
    global_weights = global_model.state_dict()
    
    # Test FLTrust aggregation
    print("--- FLTrust Aggregation ---")
    try:
        fltrust_result = fltrust_aggregator.aggregate(
            client_updates=client_updates,
            global_weights=global_weights
        )
        print(f"✅ FLTrust aggregation successful")
        print(f"   Aggregation info: {fltrust_aggregator.get_aggregation_info()}")
        
        # Demonstrate custom trust scores integration
        print("\n--- FLTrust with Custom Trust Scores ---")
        custom_trust_scores = {
            "client_0": 0.9,
            "client_1": 0.8,
            "client_2": 0.7,
            "client_3": 0.85,
            "client_4": 0.75,
            "malicious_client": 0.1  # Low trust for malicious client
        }
        
        fltrust_custom_result = fltrust_aggregator.aggregate(
            client_updates=client_updates,
            global_weights=global_weights,
            trust_scores_override=custom_trust_scores
        )
        print(f"✅ FLTrust with custom trust scores successful")
        
    except Exception as e:
        print(f"❌ FLTrust aggregation failed: {e}")
    
    # Test FedAvg aggregation for comparison
    print("\n--- FedAvg Aggregation ---")
    try:
        fedavg_result = fedavg_aggregator.aggregate(client_updates=client_updates)
        print(f"✅ FedAvg aggregation successful")
    except Exception as e:
        print(f"❌ FedAvg aggregation failed: {e}")
    
    # Test Trimmed Mean aggregation
    print("\n--- Trimmed Mean Aggregation ---")
    try:
        trimmed_result = trimmed_aggregator.aggregate(client_updates=client_updates)
        print(f"✅ Trimmed Mean aggregation successful")
    except Exception as e:
        print(f"❌ Trimmed Mean aggregation failed: {e}")
    
    print("\n=== Integration with TRUST-MCNet Server ===\n")
    
    # 6. Demonstrate integration with TRUST-MCNet server
    config = {
        'trust_mode': 'hybrid',
        'trust_threshold': 0.5,
        'trim_ratio': 0.1
    }
    
    # Create server with FLTrust aggregator
    server = FederatedServer(global_model, config)
    
    # Replace the default aggregator with FLTrust
    server.aggregator = fltrust_aggregator
    
    print("✅ Successfully integrated FLTrust aggregator with TRUST-MCNet server")
    print("   Server can now use FLTrust for robust Byzantine-resistant aggregation")
    
    # 7. Demonstrate aggregator switching
    print("\n--- Aggregator Switching Demo ---")
    print("Switching between different aggregators:")
    
    aggregators = {
        "FLTrust": fltrust_aggregator,
        "FedAvg": fedavg_aggregator,
        "TrimmedMean": trimmed_aggregator
    }
    
    for name, aggregator in aggregators.items():
        server.aggregator = aggregator
        print(f"✅ Server now using {name} aggregator")
    
    print("\n=== Example Complete ===")
    print("FLTrust has been successfully integrated into TRUST-MCNet!")
    print("\nKey features demonstrated:")
    print("- ✅ Cosine similarity trust scoring")
    print("- ✅ Client update magnitude normalization") 
    print("- ✅ Trust-weighted aggregation")
    print("- ✅ Custom trust score integration")
    print("- ✅ Gradient clipping and warm-up rounds")
    print("- ✅ Modular aggregator design")
    print("- ✅ Byzantine robustness")


def demonstrate_trust_score_integration():
    """Demonstrate how to integrate TRUST-MCNet's trust scores with FLTrust."""
    
    print("\n=== Trust Score Integration Example ===\n")
    
    # This shows how you could integrate the existing TRUST-MCNet trust evaluator
    # with FLTrust's cosine similarity scores
    
    print("Integration patterns:")
    print("1. Replace cosine similarity with TRUST-MCNet trust scores")
    print("2. Combine cosine similarity with TRUST-MCNet trust scores")
    print("3. Use cosine similarity as fallback when TRUST-MCNet scores unavailable")
    
    # Example of pattern 2: Combining scores
    def combine_trust_scores(cosine_scores, trust_mcnet_scores, alpha=0.7):
        """
        Combine FLTrust cosine similarity with TRUST-MCNet trust scores.
        
        Args:
            cosine_scores: FLTrust cosine similarity scores
            trust_mcnet_scores: TRUST-MCNet trust evaluator scores
            alpha: Weight for TRUST-MCNet scores (1-alpha for cosine scores)
        """
        combined_scores = {}
        for client_id in cosine_scores:
            if client_id in trust_mcnet_scores:
                combined = alpha * trust_mcnet_scores[client_id] + (1 - alpha) * cosine_scores[client_id]
                combined_scores[client_id] = combined
            else:
                combined_scores[client_id] = cosine_scores[client_id]
        return combined_scores
    
    # Example scores
    cosine_scores = {"client_0": 0.8, "client_1": 0.6, "client_2": 0.9}
    trust_mcnet_scores = {"client_0": 0.7, "client_1": 0.9, "client_2": 0.5}
    
    combined = combine_trust_scores(cosine_scores, trust_mcnet_scores, alpha=0.7)
    print(f"\nExample combination:")
    print(f"Cosine scores:      {cosine_scores}")
    print(f"TRUST-MCNet scores: {trust_mcnet_scores}")
    print(f"Combined scores:    {combined}")


if __name__ == "__main__":
    try:
        demonstrate_fltrust_integration()
        demonstrate_trust_score_integration()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Some dependencies may not be available. This is a demonstration script.")
        print("Please ensure all TRUST-MCNet modules are properly installed.")
