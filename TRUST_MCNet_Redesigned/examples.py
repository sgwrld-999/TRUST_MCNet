#!/usr/bin/env python3
"""
Example usage script for TRUST-MCNet redesigned framework.

This script demonstrates various ways to run the federated learning simulation
with different configurations.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("✅ Command completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    return True


def main():
    """Run example configurations."""
    print("TRUST-MCNet Redesigned - Example Usage")
    print("This script demonstrates various configuration options")
    
    # Check if we're in the right directory
    if not Path("train.py").exists():
        print("❌ Error: train.py not found. Please run this script from the TRUST_MCNet_Redesigned directory.")
        sys.exit(1)
    
    examples = [
        {
            "cmd": "python train.py --help",
            "desc": "Show help and available configuration options"
        },
        {
            "cmd": "python train.py dataset=mnist model=mlp strategy=fedavg trust=hybrid federated.num_rounds=2",
            "desc": "Basic MNIST training with MLP model, FedAvg strategy, and hybrid trust (2 rounds)"
        },
        {
            "cmd": "python train.py dataset=mnist model=lstm strategy=fedadam trust=cosine federated.num_rounds=2",
            "desc": "MNIST with LSTM model, FedAdam strategy, and cosine similarity trust"
        },
        {
            "cmd": "python train.py env=iot dataset.num_clients=3 federated.num_rounds=2",
            "desc": "IoT environment simulation with resource constraints"
        },
        {
            "cmd": "python train.py trust=entropy strategy=fedprox federated.num_rounds=2",
            "desc": "Entropy-based trust with FedProx strategy"
        }
    ]
    
    # Ask user which examples to run
    print("\nAvailable examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}")
    
    print("\nOptions:")
    print("- Enter example numbers (e.g., '1,3,5' or '1-3')")
    print("- Enter 'all' to run all examples")
    print("- Enter 'q' to quit")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'q':
        print("Exiting...")
        return
    
    # Parse choice
    if choice == 'all':
        selected_indices = list(range(len(examples)))
    else:
        selected_indices = []
        for part in choice.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start-1, end))
            else:
                selected_indices.append(int(part) - 1)
    
    # Run selected examples
    success_count = 0
    for idx in selected_indices:
        if 0 <= idx < len(examples):
            example = examples[idx]
            if run_command(example['cmd'], example['desc']):
                success_count += 1
        else:
            print(f"❌ Invalid example number: {idx + 1}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(selected_indices)} examples completed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
