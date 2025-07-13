"""
Refactored partitioning module implementing core interfaces.

This module provides production-grade data partitioning capabilities following
SOLID principles and implementing the PartitionerInterface.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod

try:
    import torch
    from torch.utils.data import Dataset, Subset
    TORCH_AVAILABLE = True
except ImportError:
    # Fallback for environments without torch
    TORCH_AVAILABLE = False
    Dataset = Any
    Subset = Any

from ..core.abstractions import BasePartitioner
from ..core.interfaces import PartitionerInterface
from ..core.types import PartitionConfig, ClientID
from ..core.exceptions import PartitioningError, ConfigurationError

logger = logging.getLogger(__name__)


class IIDPartitioner(BasePartitioner):
    """
    Independent and Identically Distributed (IID) partitioning strategy.
    
    Provides equal distribution of data among clients with random sampling
    to ensure each client gets a representative subset of the data.
    """
    
    def __init__(self, config: PartitionConfig):
        """
        Initialize IID partitioner.
        
        Args:
            config: Partition configuration containing num_clients and other settings
        """
        super().__init__(config)
        self.random_seed = config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        self.logger.info(f"Initialized IID partitioner with seed: {self.random_seed}")
    
    def partition(self, data: Any, num_clients: int, **kwargs) -> List[Any]:
        """
        Partition dataset into IID subsets.
        
        Args:
            data: Dataset to partition
            num_clients: Number of clients
            **kwargs: Additional arguments (ignored for IID)
            
        Returns:
            List of IID subsets
            
        Raises:
            PartitioningError: If partitioning fails
        """
        try:
            if not TORCH_AVAILABLE:
                raise PartitioningError("PyTorch is required for dataset partitioning")
            
            dataset_size = len(data)
            if dataset_size < num_clients:
                raise PartitioningError(
                    f"Dataset size ({dataset_size}) < num_clients ({num_clients})"
                )
            
            self.logger.info(f"Creating IID partition for {num_clients} clients")
            
            # Shuffle indices for random distribution
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            
            # Calculate split sizes with proper remainder distribution
            base_size = dataset_size // num_clients
            remainder = dataset_size % num_clients
            
            client_subsets = []
            start_idx = 0
            
            for i in range(num_clients):
                # Distribute remainder among first few clients
                client_size = base_size + (1 if i < remainder else 0)
                end_idx = start_idx + client_size
                
                client_indices = indices[start_idx:end_idx]
                client_subsets.append(Subset(data, client_indices))
                
                self.logger.debug(f"Client {i}: {len(client_indices)} samples")
                start_idx = end_idx
            
            # Validate the partition
            self.validate_partition(client_subsets)
            
            self.logger.info(f"Successfully created {len(client_subsets)} IID partitions")
            return client_subsets
            
        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            self.logger.error(f"IID partitioning failed: {e}")
            raise PartitioningError(f"IID partitioning failed: {e}") from e


class DirichletPartitioner(BasePartitioner):
    """
    Dirichlet distribution-based non-IID partitioning strategy.
    
    Creates realistic non-IID data distributions where each client
    has a different proportion of each class, following a Dirichlet distribution.
    """
    
    def __init__(self, config: PartitionConfig):
        """
        Initialize Dirichlet partitioner.
        
        Args:
            config: Partition configuration containing num_clients, alpha, and other settings
        """
        super().__init__(config)
        self.alpha = config.get('alpha', 0.5)
        self.random_seed = config.get('random_seed', 42)
        self.min_samples_per_client_per_class = config.get('min_samples_per_client_per_class', 0)
        
        if self.alpha <= 0:
            raise ConfigurationError(f"Dirichlet alpha must be > 0, got {self.alpha}")
        
        np.random.seed(self.random_seed)
        self.logger.info(f"Initialized Dirichlet partitioner with alpha: {self.alpha}")
    
    def partition(self, data: Any, num_clients: int, **kwargs) -> List[Any]:
        """
        Partition dataset using Dirichlet distribution.
        
        Args:
            data: Dataset to partition
            num_clients: Number of clients
            **kwargs: Additional arguments including 'alpha' override
            
        Returns:
            List of non-IID subsets based on Dirichlet distribution
            
        Raises:
            PartitioningError: If partitioning fails
        """
        try:
            if not TORCH_AVAILABLE:
                raise PartitioningError("PyTorch is required for dataset partitioning")
            
            # Use provided alpha or default
            alpha = kwargs.get('alpha', self.alpha)
            if alpha <= 0:
                raise PartitioningError(f"Dirichlet alpha must be > 0, got {alpha}")
            
            self.logger.info(f"Creating Dirichlet partition for {num_clients} clients with alpha={alpha}")
            
            # Extract labels from dataset
            labels = self._extract_labels_safely(data)
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            
            self.logger.info(f"Found {num_classes} classes in dataset")
            
            # Create class-wise index mapping
            indices_per_class = {
                label: np.where(labels == label)[0] 
                for label in unique_labels
            }
            
            # Generate Dirichlet proportions for each class
            client_subsets = [[] for _ in range(num_clients)]
            
            for class_label in unique_labels:
                class_indices = indices_per_class[class_label]
                np.random.shuffle(class_indices)
                
                # Sample proportions from Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # Ensure minimum samples per client if specified
                if self.min_samples_per_client_per_class > 0:
                    min_total = self.min_samples_per_client_per_class * num_clients
                    if len(class_indices) < min_total:
                        self.logger.warning(
                            f"Class {class_label} has only {len(class_indices)} samples, "
                            f"but {min_total} required for minimum allocation"
                        )
                
                # Distribute class samples among clients
                self._distribute_class_samples(
                    class_indices, proportions, client_subsets, class_label
                )
            
            # Convert to Subset objects and validate
            final_subsets = self._create_validated_subsets(data, client_subsets, num_clients)
            
            self.logger.info(f"Successfully created {len(final_subsets)} Dirichlet partitions")
            return final_subsets
            
        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            self.logger.error(f"Dirichlet partitioning failed: {e}")
            raise PartitioningError(f"Dirichlet partitioning failed: {e}") from e
    
    def _extract_labels_safely(self, dataset: Any) -> np.ndarray:
        """
        Safely extract labels from dataset with error handling.
        
        Args:
            dataset: Dataset to extract labels from
            
        Returns:
            Array of labels
            
        Raises:
            PartitioningError: If label extraction fails
        """
        labels = []
        failed_samples = 0
        
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                # Handle different label types
                if hasattr(label, 'item'):  # Tensor
                    labels.append(label.item())
                elif isinstance(label, (int, np.integer)):
                    labels.append(int(label))
                elif isinstance(label, (float, np.floating)):
                    labels.append(int(label))
                else:
                    # Try to convert to int
                    labels.append(int(label))
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract label for sample {i}: {e}")
                labels.append(0)  # Default label
                failed_samples += 1
        
        if failed_samples > 0:
            self.logger.warning(f"Failed to extract labels for {failed_samples} samples")
        
        if not labels:
            raise PartitioningError("Could not extract any labels from dataset")
        
        return np.array(labels)
    
    def _distribute_class_samples(
        self, 
        class_indices: np.ndarray, 
        proportions: np.ndarray, 
        client_subsets: List[List], 
        class_label: Any
    ) -> None:
        """Distribute samples of a single class among clients."""
        # Ensure proportions sum to 1 and handle edge cases
        proportions = np.maximum(proportions, 1e-10)
        proportions = proportions / proportions.sum()
        
        # Calculate split points
        cumulative_proportions = np.cumsum(proportions)
        split_points = (cumulative_proportions * len(class_indices)).astype(int)
        
        start_idx = 0
        for client_id in range(len(proportions)):
            end_idx = (
                split_points[client_id] 
                if client_id < len(split_points) - 1 
                else len(class_indices)
            )
            
            if end_idx > start_idx:
                client_indices = class_indices[start_idx:end_idx]
                client_subsets[client_id].extend(client_indices)
                self.logger.debug(
                    f"Client {client_id}, Class {class_label}: {len(client_indices)} samples"
                )
            
            start_idx = end_idx
    
    def _create_validated_subsets(
        self, 
        dataset: Any, 
        client_subsets: List[List], 
        num_clients: int
    ) -> List[Any]:
        """Create and validate subset objects."""
        final_subsets = []
        empty_clients = []
        
        for client_id, client_indices in enumerate(client_subsets):
            if client_indices:
                subset = Subset(dataset, client_indices)
                final_subsets.append(subset)
                self.logger.debug(f"Client {client_id}: {len(client_indices)} samples")
            else:
                empty_clients.append(client_id)
        
        # Handle empty clients
        if empty_clients:
            self.logger.warning(f"Clients {empty_clients} received no samples")
            if len(final_subsets) == 0:
                raise PartitioningError("No clients received any data")
        
        # Validate the partition
        if final_subsets:
            self.validate_partition(final_subsets)
        
        return final_subsets


class PathologicalPartitioner(BasePartitioner):
    """
    Pathological non-IID partitioning strategy.
    
    Each client receives data from only a limited number of classes,
    creating extreme non-IID conditions for federated learning research.
    """
    
    def __init__(self, config: PartitionConfig):
        """
        Initialize pathological partitioner.
        
        Args:
            config: Partition configuration containing num_clients, classes_per_client, etc.
        """
        super().__init__(config)
        self.classes_per_client = config.get('classes_per_client', 2)
        self.random_seed = config.get('random_seed', 42)
        
        if self.classes_per_client < 1:
            raise ConfigurationError(
                f"classes_per_client must be >= 1, got {self.classes_per_client}"
            )
        
        np.random.seed(self.random_seed)
        self.logger.info(
            f"Initialized pathological partitioner with {self.classes_per_client} classes per client"
        )
    
    def partition(self, data: Any, num_clients: int, **kwargs) -> List[Any]:
        """
        Partition dataset pathologically (limited classes per client).
        
        Args:
            data: Dataset to partition
            num_clients: Number of clients
            **kwargs: Additional arguments including 'classes_per_client' override
            
        Returns:
            List of pathological subsets
            
        Raises:
            PartitioningError: If partitioning fails
        """
        try:
            if not TORCH_AVAILABLE:
                raise PartitioningError("PyTorch is required for dataset partitioning")
            
            # Use provided classes_per_client or default
            classes_per_client = kwargs.get('classes_per_client', self.classes_per_client)
            if classes_per_client < 1:
                raise PartitioningError(
                    f"classes_per_client must be >= 1, got {classes_per_client}"
                )
            
            self.logger.info(
                f"Creating pathological partition for {num_clients} clients "
                f"with {classes_per_client} classes per client"
            )
            
            # Extract labels and get class information
            labels = self._extract_labels_safely(data)
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            
            if classes_per_client > num_classes:
                self.logger.warning(
                    f"classes_per_client ({classes_per_client}) > num_classes ({num_classes}), "
                    f"using {num_classes} classes per client"
                )
                classes_per_client = num_classes
            
            # Create class-wise index mapping
            indices_per_class = {
                label: np.where(labels == label)[0] 
                for label in unique_labels
            }
            
            # Assign classes to clients
            client_class_assignments = self._assign_classes_to_clients(
                unique_labels, num_clients, classes_per_client
            )
            
            # Create client subsets
            client_subsets = self._create_pathological_subsets(
                data, indices_per_class, client_class_assignments
            )
            
            self.logger.info(f"Successfully created {len(client_subsets)} pathological partitions")
            return client_subsets
            
        except Exception as e:
            if isinstance(e, PartitioningError):
                raise
            self.logger.error(f"Pathological partitioning failed: {e}")
            raise PartitioningError(f"Pathological partitioning failed: {e}") from e
    
    def _extract_labels_safely(self, dataset: Any) -> np.ndarray:
        """Safely extract labels from dataset (same as DirichletPartitioner)."""
        labels = []
        failed_samples = 0
        
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                if hasattr(label, 'item'):
                    labels.append(label.item())
                elif isinstance(label, (int, np.integer)):
                    labels.append(int(label))
                elif isinstance(label, (float, np.floating)):
                    labels.append(int(label))
                else:
                    labels.append(int(label))
            except Exception as e:
                self.logger.warning(f"Failed to extract label for sample {i}: {e}")
                labels.append(0)
                failed_samples += 1
        
        if failed_samples > 0:
            self.logger.warning(f"Failed to extract labels for {failed_samples} samples")
        
        if not labels:
            raise PartitioningError("Could not extract any labels from dataset")
        
        return np.array(labels)
    
    def _assign_classes_to_clients(
        self, 
        unique_labels: np.ndarray, 
        num_clients: int, 
        classes_per_client: int
    ) -> Dict[int, Set[int]]:
        """
        Assign classes to clients in a round-robin fashion.
        
        Args:
            unique_labels: Array of unique class labels
            num_clients: Number of clients
            classes_per_client: Number of classes per client
            
        Returns:
            Dictionary mapping client_id to set of assigned classes
        """
        client_assignments = {client_id: set() for client_id in range(num_clients)}
        
        # Shuffle classes for random assignment
        shuffled_labels = unique_labels.copy()
        np.random.shuffle(shuffled_labels)
        
        # Assign classes in round-robin fashion
        for i, label in enumerate(shuffled_labels):
            for client_id in range(num_clients):
                if len(client_assignments[client_id]) < classes_per_client:
                    client_assignments[client_id].add(label)
                    break
        
        # Ensure all clients have at least one class
        for client_id in range(num_clients):
            if len(client_assignments[client_id]) == 0:
                # Assign a random class to empty clients
                available_classes = set(unique_labels) - set().union(*client_assignments.values())
                if available_classes:
                    client_assignments[client_id].add(available_classes.pop())
                else:
                    # Reuse a class if necessary
                    client_assignments[client_id].add(np.random.choice(unique_labels))
        
        # Log assignments
        for client_id, assigned_classes in client_assignments.items():
            self.logger.debug(f"Client {client_id} assigned classes: {sorted(assigned_classes)}")
        
        return client_assignments
    
    def _create_pathological_subsets(
        self, 
        dataset: Any, 
        indices_per_class: Dict[Any, np.ndarray], 
        client_class_assignments: Dict[int, Set[int]]
    ) -> List[Any]:
        """Create pathological subsets based on class assignments."""
        client_subsets = []
        
        for client_id, assigned_classes in client_class_assignments.items():
            client_indices = []
            
            for class_label in assigned_classes:
                if class_label in indices_per_class:
                    class_indices = indices_per_class[class_label].copy()
                    np.random.shuffle(class_indices)
                    
                    # Distribute class samples among clients with this class
                    clients_with_class = [
                        cid for cid, classes in client_class_assignments.items()
                        if class_label in classes
                    ]
                    
                    if len(clients_with_class) > 1:
                        # Split class samples among multiple clients
                        client_position = clients_with_class.index(client_id)
                        split_size = len(class_indices) // len(clients_with_class)
                        start_idx = client_position * split_size
                        end_idx = (
                            start_idx + split_size 
                            if client_position < len(clients_with_class) - 1 
                            else len(class_indices)
                        )
                        client_class_indices = class_indices[start_idx:end_idx]
                    else:
                        # Client gets all samples of this class
                        client_class_indices = class_indices
                    
                    client_indices.extend(client_class_indices)
                    self.logger.debug(
                        f"Client {client_id}, Class {class_label}: {len(client_class_indices)} samples"
                    )
            
            if client_indices:
                subset = Subset(dataset, client_indices)
                client_subsets.append(subset)
                self.logger.debug(f"Client {client_id}: {len(client_indices)} total samples")
            else:
                self.logger.warning(f"Client {client_id} received no samples")
        
        # Validate the partition
        if client_subsets:
            self.validate_partition(client_subsets)
        
        return client_subsets


class PartitionerRegistry:
    """
    Registry for partitioning strategies following the Registry pattern.
    
    Allows easy extension and configuration-driven partitioner selection.
    """
    
    _partitioners: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, partitioner_class: type) -> None:
        """Register a partitioner class."""
        if not issubclass(partitioner_class, PartitionerInterface):
            raise ConfigurationError(f"Partitioner class must implement PartitionerInterface")
        
        cls._partitioners[name] = partitioner_class
        logger.info(f"Registered partitioner: {name}")
    
    @classmethod
    def get_partitioner_class(cls, name: str) -> type:
        """Get a partitioner class by name."""
        if name not in cls._partitioners:
            raise ConfigurationError(f"Unknown partitioner: {name}")
        
        return cls._partitioners[name]
    
    @classmethod
    def create_partitioner(cls, config: PartitionConfig) -> PartitionerInterface:
        """Create a partitioner instance from configuration."""
        partitioner_name = config.get('name')
        if not partitioner_name:
            raise ConfigurationError("Partitioner name not specified in config")
        
        partitioner_class = cls.get_partitioner_class(partitioner_name)
        return partitioner_class(config)
    
    @classmethod
    def list_partitioners(cls) -> List[str]:
        """List all registered partitioners."""
        return list(cls._partitioners.keys())


# Register built-in partitioners
PartitionerRegistry.register('iid', IIDPartitioner)
PartitionerRegistry.register('dirichlet', DirichletPartitioner)
PartitionerRegistry.register('pathological', PathologicalPartitioner)
