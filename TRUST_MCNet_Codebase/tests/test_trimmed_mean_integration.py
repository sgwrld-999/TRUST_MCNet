"""
Test script for coordinate-wise trimmed mean aggregation in TRUST-MCNet.

This script tests both the standalone implementation and the PyTorch integration
within the TrimmedMeanAggregator class.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TRUST_MCNet_Codebase.utils.trimmed_mean import trimmed_mean_aggregation
from TRUST_MCNet_Codebase.utils.aggregator import TrimmedMeanAggregator


def test_standalone_implementation():
    """Test the standalone trimmed mean implementation."""
    print("=== Testing Standalone Implementation ===\n")
    
    # Test case from the requirements
    client_updates = [
        [0.9, 1.0, 1.1],
        [1.0, 1.2, 1.3],
        [5.0, 0.8, 1.2],   # potentially malicious
        [1.1, 0.9, 1.0],
        [-4.0, 1.1, 1.1]   # potentially malicious
    ]
    
    print("Input client updates:")
    for i, update in enumerate(client_updates):
        print(f"  Client {i}: {update}")
    
    # Test with b=1 (trim 1 smallest and 1 largest per coordinate)
    b = 1
    result = trimmed_mean_aggregation(client_updates, b)
    print(f"\nTrimmed mean (b={b}): {result}")
    print(f"Rounded: {[round(x, 3) for x in result]}")
    
    # Verify the expected result matches the requirements
    expected = [1.0, 1.0, 1.133]  # Approximately
    for i, (actual, exp) in enumerate(zip(result, expected)):
        assert abs(actual - exp) < 0.01, f"Coordinate {i}: expected ~{exp}, got {actual}"
    
    print("✅ Standalone implementation test passed!")


def test_pytorch_integration():
    """Test the PyTorch TrimmedMeanAggregator integration."""
    print("\n=== Testing PyTorch Integration ===\n")
    
    # Create mock client updates as PyTorch tensors
    device = torch.device("cpu")
    
    # Simulate a simple 2-layer neural network
    client_updates = {
        'client_0': {
            'layer1.weight': torch.tensor([[0.9, 1.0], [1.1, 0.8]], device=device),
            'layer1.bias': torch.tensor([0.1, 0.2], device=device),
            'layer2.weight': torch.tensor([[1.0, 1.1]], device=device),
            'layer2.bias': torch.tensor([0.05], device=device)
        },
        'client_1': {
            'layer1.weight': torch.tensor([[1.0, 1.2], [0.9, 1.0]], device=device),
            'layer1.bias': torch.tensor([0.15, 0.18], device=device),
            'layer2.weight': torch.tensor([[1.2, 0.9]], device=device),
            'layer2.bias': torch.tensor([0.08], device=device)
        },
        'client_2': {  # Potentially malicious - extreme values
            'layer1.weight': torch.tensor([[5.0, -4.0], [2.0, 0.5]], device=device),
            'layer1.bias': torch.tensor([1.0, -0.5], device=device),
            'layer2.weight': torch.tensor([[3.0, 0.1]], device=device),
            'layer2.bias': torch.tensor([0.5], device=device)
        },
        'client_3': {
            'layer1.weight': torch.tensor([[1.1, 0.9], [1.0, 0.95]], device=device),
            'layer1.bias': torch.tensor([0.12, 0.22], device=device),
            'layer2.weight': torch.tensor([[0.95, 1.05]], device=device),
            'layer2.bias': torch.tensor([0.06], device=device)
        },
        'client_4': {  # Another potentially malicious client
            'layer1.weight': torch.tensor([[-2.0, 3.0], [0.1, -1.0]], device=device),
            'layer1.bias': torch.tensor([-0.3, 0.8], device=device),
            'layer2.weight': torch.tensor([[0.2, 2.0]], device=device),
            'layer2.bias': torch.tensor([-0.1], device=device)
        }
    }
    
    print("Client updates (PyTorch tensors):")
    for client_id, params in client_updates.items():
        print(f"  {client_id}:")
        for param_name, tensor in params.items():
            print(f"    {param_name}: {tensor.tolist()}")
    
    # Create trimmed mean aggregator
    aggregator = TrimmedMeanAggregator(trim_ratio=0.2, clip_threshold=10.0)  # trim 20%
    
    # Perform aggregation
    result = aggregator.aggregate(client_updates)
    
    print(f"\nAggregated result:")
    for param_name, tensor in result.items():
        print(f"  {param_name}: {tensor.tolist()}")
    
    # Verify that extreme values are handled
    # The malicious clients should have reduced impact
    layer1_weight = result['layer1.weight']
    
    # Check that the result is reasonable (not dominated by extreme values)
    max_val = torch.max(torch.abs(layer1_weight)).item()
    assert max_val < 3.0, f"Aggregated values too extreme: max={max_val}"
    
    print("✅ PyTorch integration test passed!")


def test_robustness_scenarios():
    """Test various robustness scenarios."""
    print("\n=== Testing Robustness Scenarios ===\n")
    
    # Scenario 1: All identical updates
    print("Scenario 1: All identical updates")
    identical_updates = [[1.0, 2.0, 3.0]] * 5
    result = trimmed_mean_aggregation(identical_updates, 1)
    expected = [1.0, 2.0, 3.0]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ Identical updates handled correctly")
    
    # Scenario 2: Single outlier per coordinate
    print("\nScenario 2: Single outlier per coordinate")
    outlier_updates = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [100.0, 1.0, 1.0],  # outlier in first coordinate
        [1.0, 100.0, 1.0],  # outlier in second coordinate
        [1.0, 1.0, 100.0]   # outlier in third coordinate
    ]
    result = trimmed_mean_aggregation(outlier_updates, 1)
    # Should be close to [1.0, 1.0, 1.0] with outliers trimmed
    for val in result:
        assert abs(val - 1.0) < 0.5, f"Outlier not properly handled: {result}"
    print("✅ Single outliers handled correctly")
    
    # Scenario 3: Multiple coordinated attacks
    print("\nScenario 3: Multiple coordinated attacks")
    attack_updates = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [50.0, 50.0, 50.0],  # coordinated attack
        [-50.0, -50.0, -50.0]  # coordinated attack
    ]
    result = trimmed_mean_aggregation(attack_updates, 1)
    # Should be [1.0, 1.0, 1.0] with attacks trimmed
    for val in result:
        assert abs(val - 1.0) < 0.1, f"Coordinated attack not handled: {result}"
    print("✅ Coordinated attacks handled correctly")
    
    # Scenario 4: Edge case - minimum clients
    print("\nScenario 4: Edge case - minimum clients")
    min_updates = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ]
    result = trimmed_mean_aggregation(min_updates, 1)
    expected = [2.0, 3.0]  # Middle values
    for actual, exp in zip(result, expected):
        assert abs(actual - exp) < 0.1, f"Minimum client case failed: {result}"
    print("✅ Minimum client scenario handled correctly")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===\n")
    
    test_cases = [
        {
            'name': 'Empty client list',
            'updates': [],
            'b': 1,
            'error': ValueError
        },
        {
            'name': 'Negative trimming parameter',
            'updates': [[1, 2], [3, 4]],
            'b': -1,
            'error': ValueError
        },
        {
            'name': 'Trimming parameter too large',
            'updates': [[1, 2], [3, 4]],
            'b': 2,
            'error': ValueError
        },
        {
            'name': 'Dimension mismatch',
            'updates': [[1, 2], [3, 4, 5]],
            'b': 0,
            'error': ValueError
        }
    ]
    
    for test_case in test_cases:
        try:
            trimmed_mean_aggregation(test_case['updates'], test_case['b'])
            print(f"❌ {test_case['name']}: Expected error but none occurred")
        except test_case['error']:
            print(f"✅ {test_case['name']}: Correctly caught error")
        except Exception as e:
            print(f"❌ {test_case['name']}: Wrong error type: {type(e)}")


def main():
    """Run all tests."""
    print("Starting Coordinate-wise Trimmed Mean Tests...\n")
    
    try:
        test_standalone_implementation()
        test_pytorch_integration()
        test_robustness_scenarios()
        test_error_handling()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Coordinate-wise trimmed mean implementation is working correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
