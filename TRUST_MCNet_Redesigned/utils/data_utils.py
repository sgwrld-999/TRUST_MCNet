"""
Data utilities for TRUST-MCNet federated learning framework.

This module provides:
- CSVDataset class for loading and preprocessing CSV data
- Client data splitting with edge case handling
- Train/eval data splitting
- Data validation and integrity checks
"""

import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """
    PyTorch Dataset for CSV data with preprocessing capabilities.
    
    Handles data loading, preprocessing, and conversion to torch tensors
    with robust error handling and validation.
    """
    
    def __init__(
        self,
        csv_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        preprocessing: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CSV dataset.
        
        Args:
            csv_path: Path to CSV file
            target_column: Name of target column
            feature_columns: List of feature column names (None = all except target)
            preprocessing: Dictionary of preprocessing options
        """
        self.csv_path = Path(csv_path)
        self.target_column = target_column
        self.preprocessing = preprocessing or {}
        
        # Validate file exists
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data(feature_columns)
        
        logger.info(f"Loaded CSV dataset: {len(self.data)} samples, "
                   f"{self.data.shape[1] - 1} features")
    
    def _load_and_preprocess_data(self, feature_columns: Optional[List[str]]) -> pd.DataFrame:
        """Load CSV data and apply preprocessing."""
        try:
            # Load CSV
            data = pd.read_csv(self.csv_path)
            
            # Validate target column exists
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in CSV")
            
            # Select feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != self.target_column]
            
            # Validate feature columns exist
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            
            # Select features and target
            features = data[feature_columns]
            target = data[self.target_column]
            
            # Apply preprocessing
            if self.preprocessing.get('impute_missing', False):
                features = self._impute_missing_values(features)
            
            if self.preprocessing.get('standardize', False):
                features = self._standardize_features(features)
            
            if self.preprocessing.get('encode_categoricals', False):
                features = self._encode_categorical_features(features)
            
            # Combine features and target
            processed_data = pd.concat([features, target], axis=1)
            
            # Remove any remaining NaN rows
            processed_data = processed_data.dropna()
            
            if len(processed_data) == 0:
                raise ValueError("No valid samples remaining after preprocessing")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    def _impute_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in features."""
        strategy = self.preprocessing.get('impute_strategy', 'median')
        
        # Separate numeric and categorical columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'median':
                features[numeric_cols] = features[numeric_cols].fillna(
                    features[numeric_cols].median()
                )
            elif strategy == 'mean':
                features[numeric_cols] = features[numeric_cols].fillna(
                    features[numeric_cols].mean()
                )
            else:  # mode or constant
                features[numeric_cols] = features[numeric_cols].fillna(0)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            features[categorical_cols] = features[categorical_cols].fillna(
                features[categorical_cols].mode().iloc[0]
            )
        
        return features
    
    def _standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize numeric features."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            features[numeric_cols] = (
                features[numeric_cols] - features[numeric_cols].mean()
            ) / features[numeric_cols].std()
        
        return features
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns
        
        for col in categorical_cols:
            # Simple label encoding
            unique_values = features[col].unique()
            value_to_int = {val: idx for idx, val in enumerate(unique_values)}
            features[col] = features[col].map(value_to_int)
        
        return features
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        row = self.data.iloc[idx]
        
        # Separate features and target
        features = row.drop(self.target_column).values.astype(np.float32)
        target = row[self.target_column]
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return features_tensor, target_tensor


def load_mnist_dataset(
    data_path: str = "./data/MNIST",
    binary_classification: Optional[Dict[str, Any]] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load MNIST dataset with optional binary classification setup.
    
    Args:
        data_path: Path to MNIST data directory
        binary_classification: Config for binary classification
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    
    # Apply binary classification if configured
    if binary_classification and binary_classification.get('enabled', False):
        normal_classes = binary_classification.get('normal_classes', [0, 1, 2, 3, 4, 5, 6, 8, 9])
        anomaly_classes = binary_classification.get('anomaly_classes', [7])
        
        train_dataset = _convert_to_binary_classification(train_dataset, normal_classes, anomaly_classes)
        test_dataset = _convert_to_binary_classification(test_dataset, normal_classes, anomaly_classes)
    
    logger.info(f"Loaded MNIST dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_dataset, test_dataset


def _convert_to_binary_classification(dataset: Dataset, normal_classes: List[int], anomaly_classes: List[int]) -> Dataset:
    """Convert multi-class dataset to binary classification."""
    class BinaryMNIST(Dataset):
        def __init__(self, original_dataset, normal_classes, anomaly_classes):
            self.original_dataset = original_dataset
            self.normal_classes = set(normal_classes)
            self.anomaly_classes = set(anomaly_classes)
            
            # Filter and create new indices
            self.indices = []
            for idx in range(len(original_dataset)):
                _, label = original_dataset[idx]
                if label in self.normal_classes or label in self.anomaly_classes:
                    self.indices.append(idx)
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            image, label = self.original_dataset[original_idx]
            
            # Convert to binary label (0: normal, 1: anomaly)
            binary_label = 1 if label in self.anomaly_classes else 0
            
            return image, binary_label
    
    return BinaryMNIST(dataset, normal_classes, anomaly_classes)


def split_clients(
    dataset: Dataset,
    num_clients: int,
    partitioning: str = "iid",
    dirichlet_alpha: float = 0.5
) -> List[Subset]:
    """
    Split dataset into client subsets with edge case handling.
    
    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients (must be >= 1)
        partitioning: Partitioning method ('iid', 'dirichlet', 'pathological')
        dirichlet_alpha: Alpha parameter for Dirichlet distribution
        
    Returns:
        List of dataset subsets for each client
        
    Raises:
        ValueError: If num_clients < 1 or insufficient data
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty")
    
    # Ensure each client gets at least 1 sample
    if dataset_size < num_clients:
        raise ValueError(f"Dataset size ({dataset_size}) < num_clients ({num_clients})")
    
    logger.info(f"Splitting dataset ({dataset_size} samples) into {num_clients} clients using {partitioning} partitioning")
    
    if partitioning == "iid":
        return _split_iid(dataset, num_clients)
    elif partitioning == "dirichlet":
        return _split_dirichlet(dataset, num_clients, dirichlet_alpha)
    elif partitioning == "pathological":
        return _split_pathological(dataset, num_clients)
    else:
        raise ValueError(f"Unknown partitioning method: {partitioning}")


def _split_iid(dataset: Dataset, num_clients: int) -> List[Subset]:
    """Split dataset into IID subsets."""
    dataset_size = len(dataset)
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
        
        start_idx = end_idx
        
        logger.debug(f"Client {i}: {len(client_indices)} samples")
    
    return client_subsets


def _split_dirichlet(dataset: Dataset, num_clients: int, alpha: float) -> List[Subset]:
    """Split dataset using Dirichlet distribution for non-IID partitioning."""
    # Get labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Generate Dirichlet distribution for each class
    client_subsets = []
    indices_per_class = {label: np.where(labels == label)[0] for label in unique_labels}
    
    for class_label in unique_labels:
        class_indices = indices_per_class[class_label]
        np.random.shuffle(class_indices)
        
        # Generate Dirichlet proportions
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()  # Normalize
        
        # Split class indices according to proportions
        start_idx = 0
        for i, prop in enumerate(proportions):
            end_idx = start_idx + int(prop * len(class_indices))
            if i == num_clients - 1:  # Last client gets remaining samples
                end_idx = len(class_indices)
            
            client_class_indices = class_indices[start_idx:end_idx]
            
            # Initialize client subset if not exists
            if len(client_subsets) <= i:
                client_subsets.append([])
            
            client_subsets[i].extend(client_class_indices)
            start_idx = end_idx
    
    # Convert to Subset objects and ensure each client has at least 1 sample
    final_subsets = []
    for i, client_indices in enumerate(client_subsets):
        if len(client_indices) == 0:
            # Give at least one sample to empty clients
            # Take from the largest client
            largest_client_idx = max(range(len(client_subsets)), 
                                   key=lambda x: len(client_subsets[x]))
            if len(client_subsets[largest_client_idx]) > 1:
                client_indices = [client_subsets[largest_client_idx].pop()]
        
        final_subsets.append(Subset(dataset, client_indices))
        logger.debug(f"Client {i}: {len(client_indices)} samples")
    
    return final_subsets


def _split_pathological(dataset: Dataset, num_clients: int) -> List[Subset]:
    """Split dataset pathologically (each client gets only a few classes)."""
    # Get labels from dataset
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Each client gets at most 2 classes
    classes_per_client = min(2, max(1, num_classes // num_clients))
    
    # Shuffle classes and assign to clients
    shuffled_classes = np.random.permutation(unique_labels)
    
    client_subsets = []
    for i in range(num_clients):
        start_class_idx = (i * classes_per_client) % num_classes
        end_class_idx = min(start_class_idx + classes_per_client, num_classes)
        
        client_classes = shuffled_classes[start_class_idx:end_class_idx]
        
        # Get indices for client's classes
        client_indices = []
        for class_label in client_classes:
            class_indices = np.where(labels == class_label)[0]
            client_indices.extend(class_indices)
        
        if len(client_indices) == 0:
            # Fallback: give at least one sample
            client_indices = [i % len(dataset)]
        
        client_subsets.append(Subset(dataset, client_indices))
        logger.debug(f"Client {i}: {len(client_indices)} samples from classes {client_classes}")
    
    return client_subsets


def split_train_eval(
    dataset: Dataset,
    eval_fraction: float = 0.2
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and evaluation subsets.
    
    Args:
        dataset: Dataset to split
        eval_fraction: Fraction of data for evaluation (0 < eval_fraction < 1)
        
    Returns:
        Tuple of (train_subset, eval_subset)
        
    Raises:
        ValueError: If eval_fraction is invalid or insufficient data
    """
    if not (0 < eval_fraction < 1):
        raise ValueError(f"eval_fraction must be between 0 and 1, got {eval_fraction}")
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty")
    
    # Calculate split sizes with minimum 1 sample per split
    eval_size = max(1, int(dataset_size * eval_fraction))
    train_size = dataset_size - eval_size
    
    if train_size < 1:
        raise ValueError(f"Insufficient data for train/eval split: {dataset_size} samples")
    
    # Random split
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    
    train_subset = Subset(dataset, train_indices)
    eval_subset = Subset(dataset, eval_indices)
    
    logger.debug(f"Split dataset: {len(train_subset)} train, {len(eval_subset)} eval samples")
    
    return train_subset, eval_subset


def create_data_loaders(
    train_subset: Subset,
    eval_subset: Subset,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for train and evaluation subsets.
    
    Args:
        train_subset: Training subset
        eval_subset: Evaluation subset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, eval_loader


def validate_data_splits(
    client_subsets: List[Subset],
    min_samples_per_client: int = 1,
    max_samples_per_client: Optional[int] = None
) -> bool:
    """
    Validate that data splits meet requirements.
    
    Args:
        client_subsets: List of client dataset subsets
        min_samples_per_client: Minimum samples per client
        max_samples_per_client: Maximum samples per client (None = no limit)
        
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If validation fails
    """
    if not client_subsets:
        raise ValueError("No client subsets provided")
    
    for i, subset in enumerate(client_subsets):
        subset_size = len(subset)
        
        if subset_size < min_samples_per_client:
            raise ValueError(
                f"Client {i} has {subset_size} samples, "
                f"minimum required: {min_samples_per_client}"
            )
        
        if max_samples_per_client and subset_size > max_samples_per_client:
            raise ValueError(
                f"Client {i} has {subset_size} samples, "
                f"maximum allowed: {max_samples_per_client}"
            )
    
    total_samples = sum(len(subset) for subset in client_subsets)
    logger.info(f"Data validation passed: {len(client_subsets)} clients, {total_samples} total samples")
    
    return True
