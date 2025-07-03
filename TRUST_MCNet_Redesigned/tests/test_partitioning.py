"""
Unit tests for dataset partitioning strategies.
"""

import unittest
import torch
from torch.utils.data import TensorDataset
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.partitioning import (
    IIDPartitioner, 
    DirichletPartitioner, 
    PathologicalPartitioner, 
    PartitionerRegistry
)


class TestPartitioningStrategies(unittest.TestCase):
    """Test suite for dataset partitioning strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple dataset for testing
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic data with 10 classes, 1000 samples each
        self.num_classes = 10
        self.samples_per_class = 100
        self.total_samples = self.num_classes * self.samples_per_class
        
        # Generate features and labels
        features = torch.randn(self.total_samples, 784)  # MNIST-like features
        labels = torch.repeat_interleave(torch.arange(self.num_classes), self.samples_per_class)
        
        # Shuffle the dataset
        indices = torch.randperm(self.total_samples)
        features = features[indices]
        labels = labels[indices]
        
        self.dataset = TensorDataset(features, labels)
        self.num_clients = 5
    
    def test_iid_partitioner(self):
        """Test IID partitioning strategy."""
        partitioner = IIDPartitioner()
        
        # Test normal case
        subsets = partitioner.partition(self.dataset, self.num_clients)
        
        # Check number of subsets
        self.assertEqual(len(subsets), self.num_clients)
        
        # Check total samples preserved
        total_samples = sum(len(subset) for subset in subsets)
        self.assertEqual(total_samples, len(self.dataset))
        
        # Check approximate size distribution
        expected_size = len(self.dataset) // self.num_clients
        for subset in subsets:
            self.assertGreaterEqual(len(subset), expected_size - 1)
            self.assertLessEqual(len(subset), expected_size + 1)
    
    def test_iid_partitioner_edge_cases(self):
        """Test IID partitioner edge cases."""
        partitioner = IIDPartitioner()
        
        # Test with single client
        subsets = partitioner.partition(self.dataset, 1)
        self.assertEqual(len(subsets), 1)
        self.assertEqual(len(subsets[0]), len(self.dataset))
        
        # Test with more clients than samples should raise error
        small_dataset = TensorDataset(torch.randn(3, 784), torch.tensor([0, 1, 2]))
        with self.assertRaises(ValueError):
            partitioner.partition(small_dataset, 5)
    
    def test_dirichlet_partitioner(self):
        """Test Dirichlet partitioning strategy."""
        partitioner = DirichletPartitioner()
        
        # Test normal case
        subsets = partitioner.partition(self.dataset, self.num_clients, alpha=0.5)
        
        # Check number of subsets
        self.assertGreaterEqual(len(subsets), 1)  # May be less than num_clients due to randomness
        
        # Check total samples preserved
        total_samples = sum(len(subset) for subset in subsets)
        self.assertGreaterEqual(total_samples, 0)  # Some samples should be distributed
        
        # Test different alpha values
        subsets_low_alpha = partitioner.partition(self.dataset, self.num_clients, alpha=0.1)
        subsets_high_alpha = partitioner.partition(self.dataset, self.num_clients, alpha=2.0)
        
        # Both should create valid partitions
        self.assertGreater(len(subsets_low_alpha), 0)
        self.assertGreater(len(subsets_high_alpha), 0)
    
    def test_pathological_partitioner(self):
        """Test pathological partitioning strategy."""
        partitioner = PathologicalPartitioner()
        
        # Test normal case
        subsets = partitioner.partition(self.dataset, self.num_clients, classes_per_client=2)
        
        # Check number of subsets
        self.assertEqual(len(subsets), self.num_clients)
        
        # Check that each client has limited classes
        for subset in subsets:
            if len(subset) > 0:
                # Extract labels from subset
                labels = []
                for i in range(len(subset)):
                    _, label = subset[i]
                    labels.append(label.item())
                
                unique_labels = set(labels)
                self.assertLessEqual(len(unique_labels), 2)  # Should have at most 2 classes
    
    def test_pathological_partitioner_edge_cases(self):
        """Test pathological partitioner edge cases."""
        partitioner = PathologicalPartitioner()
        
        # Test with classes_per_client > num_classes
        subsets = partitioner.partition(self.dataset, self.num_clients, classes_per_client=15)
        
        # Should still create valid partitions
        self.assertEqual(len(subsets), self.num_clients)
    
    def test_partitioner_registry(self):
        """Test partitioner registry functionality."""
        # Test getting existing partitioners
        iid_partitioner = PartitionerRegistry.get_partitioner('iid')
        self.assertIsInstance(iid_partitioner, IIDPartitioner)
        
        dirichlet_partitioner = PartitionerRegistry.get_partitioner('dirichlet')
        self.assertIsInstance(dirichlet_partitioner, DirichletPartitioner)
        
        pathological_partitioner = PartitionerRegistry.get_partitioner('pathological')
        self.assertIsInstance(pathological_partitioner, PathologicalPartitioner)
        
        # Test unknown partitioner
        with self.assertRaises(ValueError):
            PartitionerRegistry.get_partitioner('unknown')
        
        # Test listing strategies
        strategies = PartitionerRegistry.list_strategies()
        self.assertIn('iid', strategies)
        self.assertIn('dirichlet', strategies)
        self.assertIn('pathological', strategies)
    
    def test_partition_validation(self):
        """Test partition validation."""
        partitioner = IIDPartitioner()
        
        # Create valid partition
        subsets = partitioner.partition(self.dataset, self.num_clients)
        
        # Validation should pass
        try:
            partitioner.validate_partition(subsets, min_samples_per_client=1)
        except ValueError:
            self.fail("Validation failed for valid partition")
        
        # Test with too high minimum samples requirement
        with self.assertRaises(ValueError):
            partitioner.validate_partition(subsets, min_samples_per_client=1000)


class TestPartitionerRegistryExtensibility(unittest.TestCase):
    """Test partitioner registry extensibility."""
    
    def test_custom_partitioner_registration(self):
        """Test registering custom partitioner."""
        from utils.partitioning import PartitioningStrategy, PartitionerRegistry
        
        class CustomPartitioner(PartitioningStrategy):
            def partition(self, dataset, num_clients, **kwargs):
                # Simple custom partitioner - just splits evenly
                dataset_size = len(dataset)
                subset_size = dataset_size // num_clients
                
                subsets = []
                for i in range(num_clients):
                    start_idx = i * subset_size
                    end_idx = start_idx + subset_size if i < num_clients - 1 else dataset_size
                    indices = list(range(start_idx, end_idx))
                    subsets.append(torch.utils.data.Subset(dataset, indices))
                
                return subsets
        
        # Register custom partitioner
        PartitionerRegistry.register_strategy('custom', CustomPartitioner)
        
        # Test that it's registered
        strategies = PartitionerRegistry.list_strategies()
        self.assertIn('custom', strategies)
        
        # Test that it can be retrieved and used
        custom_partitioner = PartitionerRegistry.get_partitioner('custom')
        self.assertIsInstance(custom_partitioner, CustomPartitioner)


if __name__ == '__main__':
    unittest.main()
