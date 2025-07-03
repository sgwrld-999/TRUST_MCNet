"""
Federated Learning Dataset Management

This module handles dataset preparation and partitioning for federated learning simulations.
It provides functionality to split datasets across multiple clients while supporting both
IID (Independent and Identically Distributed) and non-IID data distributions.

The module is crucial for simulating realistic federated learning scenarios where:
1. Data is distributed across multiple clients (devices/organizations)
2. Each client only has access to its local data partition
3. Data distribution can be heterogeneous (non-IID) to simulate real-world conditions

Key Features:
- MNIST dataset loading with standard preprocessing
- IID partitioning: Equal and random distribution across clients
- Non-IID partitioning: Dirichlet distribution for realistic heterogeneity
- Automatic train/validation splitting for each client
- Configurable batch sizes and validation ratios

The non-IID partitioning using Dirichlet distribution is particularly important as it
simulates real-world scenarios where different clients have different data distributions
(e.g., different users have different writing styles, different hospitals have different
patient populations).

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

class DataManager:
    """
    Dataset Manager for Federated Learning Data Partitioning
    
    This class manages the entire data pipeline for federated learning simulations,
    including dataset loading, preprocessing, partitioning across clients, and
    DataLoader creation for training and validation.
    
    Why this class exists:
    - Centralizes all data management logic for FL simulations
    - Provides consistent data preprocessing across all clients
    - Implements multiple partitioning strategies (IID vs non-IID)
    - Ensures reproducible data splits for experimental consistency
    - Simplifies data distribution complexity in federated settings
    
    How it works:
    1. Loads and preprocesses the MNIST dataset
    2. Partitions data across specified number of clients
    3. Creates train/validation splits for each client partition
    4. Generates DataLoaders for efficient batch processing
    5. Provides centralized test set for global evaluation
    
    The class supports two partitioning strategies:
    - IID: Random uniform distribution (unrealistic but useful for baselines)
    - Dirichlet: Non-IID distribution reflecting real-world heterogeneity
    """
    
    def __init__(self, cfg):
        """
        Initialize the DataManager with configuration parameters.
        
        Parameters:
        cfg: Configuration object containing dataset parameters:
            - dataset.data_path: Path to store/load dataset files
            - dataset.batch_size: Batch size for DataLoader creation
            - dataset.val_ratio: Proportion of data used for validation (0.0-1.0)
            - dataset.partitioning: Partitioning strategy ('iid' or 'dirichlet')
            - dataset.dirichlet_alpha: Alpha parameter for Dirichlet distribution
        
        Use of parameters in simulation:
        - data_path: Determines where MNIST data is stored/cached
        - batch_size: Controls memory usage and gradient noise in training
        - val_ratio: Balances training data vs validation monitoring
        - partitioning: Determines data heterogeneity across clients
        - dirichlet_alpha: Controls degree of non-IID-ness (lower = more heterogeneous)
        """
        # Store configuration parameters for dataset management
        self.data_path = cfg.dataset.data_path              # Path for dataset storage
        self.batch_size = cfg.dataset.batch_size            # Batch size for DataLoaders
        self.val_ratio = cfg.dataset.val_ratio              # Validation split ratio
        self.partitioning = cfg.dataset.partitioning        # IID or Dirichlet partitioning
        self.alpha = cfg.dataset.dirichlet_alpha            # Dirichlet concentration parameter

    def get_mnist(self):
        """
        Load and preprocess the MNIST dataset.
        
        This method handles MNIST dataset loading with standard preprocessing
        transformations. The preprocessing includes normalization with MNIST-specific
        mean and standard deviation values for optimal model performance.
        
        Why this preprocessing:
        - ToTensor(): Converts PIL images to PyTorch tensors and scales to [0,1]
        - Normalize(): Standardizes pixel values using MNIST dataset statistics
        - Mean=0.1307, Std=0.3081: Standard MNIST normalization parameters
        
        How it works:
        - Downloads MNIST data if not already present at data_path
        - Applies consistent transformations to both train and test sets
        - Returns separate train and test datasets for further processing
        
        Returns:
        tuple: (train_dataset, test_dataset) - Preprocessed MNIST datasets
        
        Use in simulation:
        - Provides standardized input data for all federated learning clients
        - Ensures consistent preprocessing across distributed training
        - Enables reproducible results across different runs
        """
        # Define preprocessing transformations for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),                           # Convert to tensor, scale to [0,1]
            transforms.Normalize((0.1307,), (0.3081,))      # Normalize with MNIST statistics
        ])
        
        # Load training and test datasets with preprocessing
        train = datasets.MNIST(self.data_path, train=True, download=True, transform=transform)
        test = datasets.MNIST(self.data_path, train=False, download=True, transform=transform)
        
        return train, test

    def partition(self, train_set, num_clients):
        """
        Partition the training dataset across federated learning clients.
        
        This method implements two partitioning strategies to simulate different
        federated learning scenarios:
        
        1. IID (Independent and Identically Distributed):
           - Randomly distributes data equally across clients
           - Each client gets similar data distribution
           - Unrealistic but useful for baselines and debugging
        
        2. Dirichlet (Non-IID):
           - Uses Dirichlet distribution to create heterogeneous partitions
           - Each client gets different class distributions
           - Simulates real-world federated learning challenges
           - Alpha parameter controls heterogeneity level
        
        Why different partitioning strategies:
        - IID: Provides baseline performance and debugging capability
        - Non-IID: Reflects real-world data heterogeneity challenges
        - Dirichlet: Mathematically principled way to control heterogeneity
        
        How Dirichlet partitioning works:
        1. For each class, sample proportions from Dirichlet distribution
        2. Split class samples according to these proportions
        3. Assign splits to different clients
        4. Combine all class splits for each client
        
        Parameters:
        train_set: PyTorch dataset to partition
        num_clients: Number of federated learning clients
        
        Returns:
        List[Dataset]: List of dataset partitions, one per client
        
        Use of parameters in simulation:
        - train_set: Source data to be distributed across clients
        - num_clients: Determines how many ways to split the data
        - self.alpha: Controls non-IID-ness (low alpha = more heterogeneous)
        """
        if self.partitioning == "iid":
            # IID Partitioning: Equal random distribution
            lengths = [len(train_set) // num_clients] * num_clients
            lengths[-1] += len(train_set) - sum(lengths)  # Handle remainder
            splits = random_split(train_set, lengths)
            
        elif self.partitioning == "dirichlet":
            # Non-IID Dirichlet Partitioning
            data_indices = [[] for _ in range(num_clients)]  # Indices for each client
            targets = np.array(train_set.targets)            # All target labels
            num_classes = len(np.unique(targets))            # Number of classes (10 for MNIST)
            
            # Process each class separately
            for c in range(num_classes):
                # Get indices for current class
                idx_k = np.where(targets == c)[0]
                np.random.shuffle(idx_k)  # Randomize order
                
                # Sample proportions from Dirichlet distribution
                # Lower alpha = more heterogeneous distribution
                proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
                
                # Calculate split points based on proportions
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                # Split class indices according to proportions
                splits_k = np.split(idx_k, split_points)
                
                # Assign splits to clients
                for client_id, idx in enumerate(splits_k):
                    data_indices[client_id] += idx.tolist()
                    
            # Create dataset subsets for each client
            splits = [Subset(train_set, inds) for inds in data_indices]
            
        else:
            raise ValueError(f"Unknown partitioning: {self.partitioning}")
            
        return splits

    def create_loaders(self, splits):
        """
        Create training and validation DataLoaders for each client partition.
        
        This method splits each client's data partition into training and validation
        sets, then creates DataLoaders for efficient batch processing during
        federated learning training and evaluation.
        
        Why separate train/validation per client:
        - Enables local validation without data sharing
        - Provides performance monitoring for each client
        - Supports early stopping and hyperparameter tuning
        - Maintains data privacy by keeping validation local
        
        How it works:
        1. For each client partition, split into train/validation
        2. Create DataLoader for training data (shuffled for better training)
        3. Create DataLoader for validation data (not shuffled for consistency)
        4. Return lists of DataLoaders aligned with client indices
        
        Parameters:
        splits: List of dataset partitions, one per client
        
        Returns:
        tuple: (train_loaders, val_loaders) - Lists of DataLoaders for each client
        
        Use in simulation:
        - train_loaders: Used for local model training on each client
        - val_loaders: Used for local performance evaluation
        - Batch processing enables efficient GPU utilization
        - Shuffling in training improves gradient diversity
        """
        train_loaders, val_loaders = [], []
        
        # Create train/validation splits and DataLoaders for each client
        for subset in splits:
            # Calculate validation set size based on configured ratio
            val_size = int(len(subset) * self.val_ratio)
            train_size = len(subset) - val_size
            
            # Split client's data into training and validation
            train_sub, val_sub = random_split(subset, [train_size, val_size])
            
            # Create DataLoaders with appropriate settings
            train_loader = DataLoader(
                train_sub, 
                batch_size=self.batch_size, 
                shuffle=True    # Shuffle for better training dynamics
            )
            val_loader = DataLoader(
                val_sub, 
                batch_size=self.batch_size, 
                shuffle=False   # No shuffle for consistent validation
            )
            
            # Store DataLoaders for this client
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            
        return train_loaders, val_loaders

    def create_test_loader(self, test_set):
        """
        Create a centralized test DataLoader for global model evaluation.
        
        This method creates a DataLoader for the test set that can be used
        for centralized evaluation of the global federated learning model.
        Unlike training data, the test set is not partitioned and represents
        a common evaluation benchmark.
        
        Why centralized test set:
        - Provides unbiased evaluation of global model performance
        - Enables comparison across different FL configurations
        - Represents real-world deployment scenario evaluation
        - Maintains consistency with traditional ML evaluation practices
        
        Parameters:
        test_set: PyTorch dataset containing test samples
        
        Returns:
        DataLoader: Test data loader for batch processing
        
        Use in simulation:
        - Global performance evaluation after FL training
        - Comparison with centralized training baselines
        - Final model assessment and reporting
        """
        return DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

    def prepare(self, num_clients):
        """
        Complete data preparation pipeline for federated learning simulation.
        
        This method orchestrates the entire data preparation process, from
        dataset loading to final DataLoader creation. It serves as the main
        entry point for data preparation in federated learning simulations.
        
        Why this method exists:
        - Provides a single interface for complete data preparation
        - Ensures consistent data processing pipeline
        - Simplifies usage in main simulation code
        - Handles all data preparation steps in correct order
        
        How it works:
        1. Load and preprocess MNIST dataset
        2. Partition training data across federated clients
        3. Create train/validation splits for each client
        4. Generate DataLoaders for efficient processing
        5. Return all necessary data components
        
        Parameters:
        num_clients: Number of federated learning clients to create data for
        
        Returns:
        tuple: (train_loaders, val_loaders, test_loader)
            - train_loaders: List of training DataLoaders, one per client
            - val_loaders: List of validation DataLoaders, one per client  
            - test_loader: Centralized test DataLoader for global evaluation
        
        Use of parameters in simulation:
        - num_clients: Determines data partitioning and number of participants
        - Returned loaders enable distributed training and evaluation
        - Each client gets its own training/validation data partition
        - Test loader provides common evaluation benchmark
        """
        # Load and preprocess datasets
        train_set, test_set = self.get_mnist()
        
        # Partition training data across clients
        splits = self.partition(train_set, num_clients)
        
        # Create DataLoaders for training and validation
        train_loaders, val_loaders = self.create_loaders(splits)
        
        # Create centralized test loader
        test_loader = self.create_test_loader(test_set)
        
        return train_loaders, val_loaders, test_loader