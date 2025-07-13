"""
Dataset partitioning strategies for federated learning.

This module implements the strategy pattern for dataset partitioning,
supporting various federated data distribution methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import logging

logger = logging.getLogger(__name__)


class PartitioningStrategy(ABC):
    """Abstract base class for dataset partitioning strategies."""
    
    @abstractmethod
    def partition(self, dataset: Dataset, num_clients: int, **kwargs) -> List[Subset]:
        """
        Partition dataset into client subsets.
        
        Args:
            dataset: PyTorch dataset to partition
            num_clients: Number of clients
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of dataset subsets for each client
        """
        pass
    
    def validate_partition(self, subsets: List[Subset], min_samples_per_client: int = 1) -> None:
        """
        Validate partition results.
        
        Args:
            subsets: List of client subsets
            min_samples_per_client: Minimum samples required per client
            
        Raises:
            ValueError: If partition is invalid
        """
        if not subsets:
            raise ValueError("No subsets created")
        
        for i, subset in enumerate(subsets):
            if len(subset) < min_samples_per_client:
                raise ValueError(
                    f"Client {i} has only {len(subset)} samples, "
                    f"minimum required: {min_samples_per_client}"
                )


class IIDPartitioner(PartitioningStrategy):
    """Independent and Identically Distributed (IID) partitioning strategy."""
    
    def partition(self, dataset: Dataset, num_clients: int, **kwargs) -> List[Subset]:
        """
        Partition dataset into IID subsets.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            
        Returns:
            List of IID subsets
        """
        dataset_size = len(dataset)
        if dataset_size < num_clients:
            raise ValueError(f"Dataset size ({dataset_size}) < num_clients ({num_clients})")
        
        logger.info(f"Creating IID partition for {num_clients} clients")
        
        # Shuffle indices for random distribution
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        base_size = dataset_size // num_clients
        remainder = dataset_size % num_clients
        
        client_subsets = []
        start_idx = 0
        
        for i in range(num_clients):
            # Distribute remainder among first few clients
            client_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + client_size
            
            client_indices = indices[start_idx:end_idx]
            client_subsets.append(Subset(dataset, client_indices))
            
            logger.debug(f"Client {i}: {len(client_indices)} samples")
            start_idx = end_idx
        
        self.validate_partition(client_subsets)
        return client_subsets


class DirichletPartitioner(PartitioningStrategy):
    """Dirichlet distribution-based non-IID partitioning strategy."""
    
    def partition(self, dataset: Dataset, num_clients: int, **kwargs) -> List[Subset]:
        """
        Partition dataset using Dirichlet distribution.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter (default: 0.5)
            
        Returns:
            List of non-IID subsets based on Dirichlet distribution
        """
        alpha = kwargs.get('alpha', 0.5)
        
        logger.info(f"Creating Dirichlet partition for {num_clients} clients with alpha={alpha}")
        
        # Extract labels from dataset
        labels = self._extract_labels(dataset)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Create class-wise index mapping
        indices_per_class = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Generate Dirichlet proportions for each class
        client_subsets = [[] for _ in range(num_clients)]
        
        for class_label in unique_labels:
            class_indices = indices_per_class[class_label]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Ensure each client gets at least one sample if possible
            proportions = np.maximum(proportions, 1e-10)
            proportions = proportions / proportions.sum()
            
            # Distribute class samples among clients
            cumulative_proportions = np.cumsum(proportions)
            split_points = (cumulative_proportions * len(class_indices)).astype(int)
            
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = split_points[client_id] if client_id < len(split_points) - 1 else len(class_indices)
                
                if end_idx > start_idx:
                    client_indices = class_indices[start_idx:end_idx]
                    client_subsets[client_id].extend(client_indices)
                
                start_idx = end_idx
        
        # Convert to Subset objects
        final_subsets = []
        for client_id, client_indices in enumerate(client_subsets):
            if client_indices:  # Only create subset if there are indices
                subset = Subset(dataset, client_indices)
                final_subsets.append(subset)
                logger.debug(f"Client {client_id}: {len(client_indices)} samples")
            else:
                logger.warning(f"Client {client_id} received no samples")
        
        if len(final_subsets) < num_clients:
            logger.warning(f"Only {len(final_subsets)} out of {num_clients} clients received data")
        
        self.validate_partition(final_subsets)
        return final_subsets
    
    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        """Extract labels from dataset."""
        labels = []
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                labels.append(label if isinstance(label, int) else label.item())
            except Exception as e:
                logger.warning(f"Failed to extract label for sample {i}: {e}")
                labels.append(0)  # Default label
        
        return np.array(labels)


class PathologicalPartitioner(PartitioningStrategy):
    """Pathological non-IID partitioning strategy (each client gets limited classes)."""
    
    def partition(self, dataset: Dataset, num_clients: int, **kwargs) -> List[Subset]:
        """
        Partition dataset pathologically (limited classes per client).
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            classes_per_client: Number of classes per client (default: 2)
            
        Returns:
            List of pathological subsets
        """
        classes_per_client = kwargs.get('classes_per_client', 2)
        
        logger.info(f"Creating pathological partition for {num_clients} clients "
                   f"with {classes_per_client} classes per client")
        
        # Extract labels and create class mapping
        labels = self._extract_labels(dataset)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        if classes_per_client > num_classes:
            logger.warning(f"classes_per_client ({classes_per_client}) > num_classes ({num_classes}), "
                          f"setting to {num_classes}")
            classes_per_client = num_classes
        
        indices_per_class = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Assign classes to clients
        client_subsets = []
        
        for client_id in range(num_clients):
            # Select classes for this client (cyclically)
            start_class_idx = (client_id * classes_per_client) % num_classes
            client_classes = []
            
            for i in range(classes_per_client):
                class_idx = (start_class_idx + i) % num_classes
                client_classes.append(unique_labels[class_idx])
            
            # Collect indices for client's classes
            client_indices = []
            for class_label in client_classes:
                class_indices = indices_per_class[class_label]
                # Distribute class samples among clients that have this class
                clients_with_class = []
                for cid in range(num_clients):
                    cid_start = (cid * classes_per_client) % num_classes
                    cid_classes = [(cid_start + j) % num_classes for j in range(classes_per_client)]
                    if (class_label == unique_labels).nonzero()[0][0] in cid_classes:
                        clients_with_class.append(cid)
                
                # Split class samples among relevant clients
                samples_per_client = len(class_indices) // len(clients_with_class)
                remainder = len(class_indices) % len(clients_with_class)
                
                client_position = clients_with_class.index(client_id)
                start_idx = client_position * samples_per_client
                end_idx = start_idx + samples_per_client + (1 if client_position < remainder else 0)
                
                client_indices.extend(class_indices[start_idx:end_idx])
            
            if client_indices:
                subset = Subset(dataset, client_indices)
                client_subsets.append(subset)
                logger.debug(f"Client {client_id}: {len(client_indices)} samples from classes {client_classes}")
        
        self.validate_partition(client_subsets)
        return client_subsets
    
    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        """Extract labels from dataset."""
        labels = []
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                labels.append(label if isinstance(label, int) else label.item())
            except Exception as e:
                logger.warning(f"Failed to extract label for sample {i}: {e}")
                labels.append(0)  # Default label
        
        return np.array(labels)


class PartitionerRegistry:
    """Registry for dataset partitioning strategies."""
    
    _strategies = {
        'iid': IIDPartitioner,
        'dirichlet': DirichletPartitioner,
        'pathological': PathologicalPartitioner
    }
    
    @classmethod
    def get_partitioner(cls, strategy_name: str) -> PartitioningStrategy:
        """
        Get partitioner instance by name.
        
        Args:
            strategy_name: Name of partitioning strategy
            
        Returns:
            Partitioner instance
            
        Raises:
            ValueError: If strategy is not registered
        """
        if strategy_name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(f"Unknown partitioning strategy: {strategy_name}. "
                           f"Available strategies: {available_strategies}")
        
        return cls._strategies[strategy_name]()
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """
        Register a new partitioning strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        if not issubclass(strategy_class, PartitioningStrategy):
            raise ValueError("Strategy class must inherit from PartitioningStrategy")
        
        cls._strategies[name] = strategy_class
        logger.info(f"Registered partitioning strategy: {name}")
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies."""
        return list(cls._strategies.keys())
