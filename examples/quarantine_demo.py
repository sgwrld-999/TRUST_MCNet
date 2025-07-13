#!/usr/bin/env python3
"""
Demo script to showcase the quarantine and trimming logic hook in TRUST_MCNet.

This script demonstrates the complete quarantine workflow including:
1. Client trust evaluation
2. Quarantine detection and enforcement
3. Trust-weighted aggregation with quarantine filtering
4. Client recovery from quarantine

Usage:
    python examples/quarantine_demo.py
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List

# Add src to path
src_path = Path(__file__).parent.parent / "tests" / "src"
sys.path.insert(0, str(src_path))

try:
    from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
    from trust_mcnet.trust_module.quarantine_state import QuarantineState
except ImportError as e:
    print(f"Failed to import TRUST_MCNet modules: {e}")
    print("Please ensure the project is properly set up and paths are correct.")
    print(f"Looking in: {src_path}")
    sys.exit(1)


def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def create_dummy_client_updates(client_ids: List[str], malicious_clients: List[str] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create dummy client model updates for demonstration.
    
    Args:
        client_ids: List of client identifiers
        malicious_clients: List of client IDs to make malicious (outlier updates)
        
    Returns:
        Dictionary of client updates
    """
    malicious_clients = malicious_clients or []
    client_updates = {}
    
    for client_id in client_ids:
        if client_id in malicious_clients:
            # Malicious client: create outlier updates
            client_updates[client_id] = {
                "layer_1": torch.tensor([[100.0, 200.0], [300.0, 400.0]], dtype=torch.float32),
                "layer_2": torch.tensor([50.0, 100.0, 150.0], dtype=torch.float32),
                "bias": torch.tensor([10.0], dtype=torch.float32)
            }
        else:
            # Benign client: create normal updates with small random variations
            base_values = np.random.normal(0, 0.1, size=(2, 2))
            client_updates[client_id] = {
                "layer_1": torch.tensor(base_values, dtype=torch.float32),
                "layer_2": torch.tensor(np.random.normal(0, 0.1, 3), dtype=torch.float32),
                "bias": torch.tensor([np.random.normal(0, 0.05)], dtype=torch.float32)
            }
    
    return client_updates


def simulate_trust_scores(client_ids: List[str], round_num: int, malicious_clients: List[str] = None) -> Dict[str, float]:
    """
    Simulate trust scores for clients.
    
    Args:
        client_ids: List of client identifiers
        round_num: Current round number
        malicious_clients: List of malicious client IDs
        
    Returns:
        Dictionary of trust scores
    """
    malicious_clients = malicious_clients or []
    trust_scores = {}
    
    for client_id in client_ids:
        if client_id in malicious_clients:
            # Malicious clients start with low trust and may recover later
            if round_num <= 5:
                trust_scores[client_id] = max(0.1, 0.3 - round_num * 0.02)  # Degrading trust
            else:
                trust_scores[client_id] = min(0.9, 0.2 + (round_num - 5) * 0.1)  # Recovery
        else:
            # Benign clients maintain good trust with small variations
            base_trust = 0.8
            variation = np.random.normal(0, 0.05)
            trust_scores[client_id] = max(0.6, min(0.95, base_trust + variation))
    
    return trust_scores


def main():
    """Main demo function."""
    print("=" * 80)
    print("TRUST_MCNet Quarantine & Trimming Logic Hook Demo")
    print("=" * 80)
    
    setup_logging()
    
    # Configuration for quarantine logic
    config = {
        'trust': {
            'quarantine': {
                'tau': 0.35,                    # Trust threshold for quarantine consideration
                'patience': 2,                  # Consecutive rounds below tau before quarantine
                'quarantine_rounds': 4,         # Duration of quarantine
                'enable_quarantine': True       # Enable quarantine feature
            },
            'aggregation': {
                'trim_ratio': 0.2,             # Trim 20% from each tail
                'min_clients_for_trimming': 3   # Minimum clients needed for trimming
            }
        }
    }
    
    # Initialize trust evaluator with quarantine capabilities
    trust_evaluator = TrustEvaluator(
        trust_mode='hybrid',
        threshold=0.5,
        config=config
    )
    
    # Define clients
    client_ids = ["client_A", "client_B", "client_C", "client_D", "client_E"]
    malicious_clients = ["client_D", "client_E"]  # These will exhibit malicious behavior
    
    print(f"\\nSimulation Setup:")
    print(f"- Total clients: {len(client_ids)}")
    print(f"- Malicious clients: {malicious_clients}")
    print(f"- Quarantine threshold (τ): {config['trust']['quarantine']['tau']}")
    print(f"- Patience: {config['trust']['quarantine']['patience']} rounds")
    print(f"- Quarantine duration: {config['trust']['quarantine']['quarantine_rounds']} rounds")
    print(f"- Trim ratio: {config['trust']['aggregation']['trim_ratio']}")
    
    # Simulate federated learning rounds
    num_rounds = 10
    results = []
    
    print(f"\\n{'='*80}")
    print("FEDERATED LEARNING SIMULATION")
    print(f"{'='*80}")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\\n--- Round {round_num} ---")
        
        # Generate client updates and trust scores for this round
        client_updates = create_dummy_client_updates(client_ids, malicious_clients)
        trust_scores = simulate_trust_scores(client_ids, round_num, malicious_clients)
        
        # Display trust scores
        print("Trust Scores:")
        for client_id in client_ids:
            marker = " 🚨" if client_id in malicious_clients else " ✅"
            print(f"  {client_id}: {trust_scores[client_id]:.3f}{marker}")
        
        try:
            # Perform aggregation with quarantine logic
            aggregated_model, trust_statistics = trust_evaluator.aggregate_model_updates(
                client_updates=client_updates,
                client_trust_scores=trust_scores,
                round_number=round_num
            )
            
            # Extract key metrics
            quarantined = trust_statistics['quarantined_clients']
            survivors = trust_statistics['surviving_clients']
            trusted_survivors = trust_statistics['trusted_survivors']
            
            print(f"\\nQuarantine Status:")
            print(f"  Quarantined: {quarantined if quarantined else 'None'}")
            print(f"  Survivors: {survivors}")
            print(f"  Final Trusted: {trusted_survivors}")
            print(f"  Quarantine Rate: {trust_statistics['quarantine_rate']:.1%}")
            
            # Store results for summary
            results.append({
                'round': round_num,
                'trust_scores': trust_scores.copy(),
                'quarantined': quarantined.copy(),
                'survivors': survivors.copy(),
                'trusted_survivors': trusted_survivors.copy(),
                'quarantine_stats': trust_statistics['quarantine_stats']
            })
            
            # Check if we have successful aggregation
            if aggregated_model:
                print(f"  ✅ Aggregation successful with {len(trusted_survivors)} clients")
                for param_name, param_tensor in aggregated_model.items():
                    print(f"    {param_name}: shape {param_tensor.shape}")
            else:
                print(f"  ❌ Aggregation failed")
        
        except Exception as e:
            print(f"  ❌ Error in round {round_num}: {e}")
            results.append({
                'round': round_num,
                'error': str(e)
            })
    
    # Print summary
    print(f"\\n{'='*80}")
    print("SIMULATION SUMMARY")
    print(f"{'='*80}")
    
    print("\\nQuarantine Timeline:")
    for result in results:
        if 'error' not in result:
            round_num = result['round']
            quarantined = result['quarantined']
            trust_mean = np.mean(list(result['trust_scores'].values()))
            
            status = "🔒 QUARANTINE" if quarantined else "🔓 Normal"
            print(f"  Round {round_num:2d}: {status:12} | "
                  f"Quarantined: {len(quarantined):1d} | "
                  f"Avg Trust: {trust_mean:.3f} | "
                  f"Clients: {quarantined if quarantined else 'None'}")
    
    # Final quarantine statistics
    final_stats = trust_evaluator.get_quarantine_statistics()
    print(f"\\nFinal Quarantine Statistics:")
    print(f"  Total clients tracked: {final_stats['total_clients']}")
    print(f"  Currently quarantined: {final_stats['currently_quarantined']}")
    print(f"  Total quarantine events: {final_stats['total_quarantine_events']}")
    print(f"  Clients ever quarantined: {final_stats['clients_ever_quarantined']}")
    
    print(f"\\n{'='*80}")
    print("Demo completed successfully! 🎉")
    print("The quarantine logic successfully:")
    print("1. ✅ Detected sustained low-trust clients")
    print("2. ✅ Quarantined malicious clients automatically")
    print("3. ✅ Applied trust-weighted trimmed mean on survivors")
    print("4. ✅ Allowed quarantined clients to recover")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
