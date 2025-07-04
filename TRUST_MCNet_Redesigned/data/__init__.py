"""
Refactored data loading module implementing core interfaces.

This module provides production-grade data loading capabilities following
SOLID principles and implementing the DataLoaderInterface.
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

from ..core.abstractions import BaseDataLoader
from ..core.interfaces import DataLoaderInterface
from ..core.types import DatasetInfo, ClientConfig
from ..core.exceptions import DataLoadingError, ConfigurationError

logger = logging.getLogger(__name__)


class TensorDataset(Dataset):
    """
    Enhanced PyTorch Dataset for tensor data with validation.
    
    Provides robust tensor handling with proper type checking and validation.
    """
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Initialize tensor dataset.
        
        Args:
            features: Feature tensor of shape (N, D)
            targets: Target tensor of shape (N,)
            
        Raises:
            DataLoadingError: If tensors have incompatible shapes
        """
        if len(features) != len(targets):
            raise DataLoadingError(
                f"Features and targets must have same length: "
                f"got {len(features)} vs {len(targets)}"
            )
        
        self.features = features
        self.targets = targets
        
        # Validate tensor types
        if not isinstance(features, torch.Tensor):
            raise DataLoadingError("Features must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise DataLoadingError("Targets must be a torch.Tensor")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        return self.features[idx], self.targets[idx]


class CSVDataLoader(BaseDataLoader):
    """
    Production-grade CSV data loader implementing DataLoaderInterface.
    
    Features:
    - Robust error handling and validation
    - Configurable preprocessing pipeline
    - Memory-efficient loading for large datasets
    - Type safety and proper logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV data loader.
        
        Args:
            config: Configuration dictionary containing:
                - path: Path to CSV file
                - target_column: Name of target column
                - feature_columns: Optional list of feature columns
                - preprocessing: Preprocessing options
                - label_mapping: Optional label mapping for classification
        """
        super().__init__(config)
        
        # Validate required CSV-specific config
        required_csv_keys = ['path', 'target_column']
        for key in required_csv_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required CSV config key: {key}")
        
        self.csv_path = Path(self.config['path'])
        self.target_column = self.config['target_column']
        self.feature_columns = self.config.get('feature_columns')
        self.preprocessing = self.config.get('preprocessing', {})
        self.label_mapping = self.config.get('label_mapping', {})
        
        # Validate file exists
        if not self.csv_path.exists():
            raise DataLoadingError(f"CSV file not found: {self.csv_path}")
        
        self.logger.info(f"Initialized CSV data loader for: {self.csv_path}")
    
    def load_data(self) -> Tuple[TensorDataset, Optional[TensorDataset]]:
        """
        Load and preprocess CSV data.
        
        Returns:
            Tuple of (train_dataset, test_dataset). Test dataset is None for CSV.
            
        Raises:
            DataLoadingError: If data loading or preprocessing fails
        """
        try:
            self.logger.info(f"Loading CSV data from: {self.csv_path}")
            
            # Load CSV
            data = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded CSV with shape: {data.shape}")
            
            # Validate and process data
            processed_data = self._preprocess_data(data)
            
            # Convert to tensors
            features, targets = self._convert_to_tensors(processed_data)
            
            # Create dataset
            dataset = TensorDataset(features, targets)
            
            self.logger.info(f"Created dataset with {len(dataset)} samples")
            return dataset, None
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            raise DataLoadingError(f"CSV data loading failed: {e}") from e
    
    def get_data_info(self) -> DatasetInfo:
        """
        Get information about the dataset.
        
        Returns:
            DatasetInfo with dataset characteristics
        """
        try:
            # Read a small sample to get info without loading full dataset
            sample_data = pd.read_csv(self.csv_path, nrows=100)
            
            # Process sample to get shape info
            processed_sample = self._preprocess_data(sample_data)
            
            num_features = len(processed_sample.columns) - 1  # Exclude target column
            num_classes = len(processed_sample[self.target_column].unique())
            
            # Get full dataset size
            full_data = pd.read_csv(self.csv_path)
            dataset_size = len(full_data)
            
            return {
                'name': self.config['name'],
                'size': dataset_size,
                'num_features': num_features,
                'num_classes': num_classes,
                'data_shape': (dataset_size, num_features),
                'target_column': self.target_column,
                'feature_columns': list(processed_sample.columns)[:-1]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get data info: {e}")
            raise DataLoadingError(f"Could not get dataset info: {e}") from e
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw pandas DataFrame
            
        Returns:
            Processed pandas DataFrame
            
        Raises:
            DataLoadingError: If preprocessing fails
        """
        try:
            # Validate target column exists
            if self.target_column not in data.columns:
                raise DataLoadingError(f"Target column '{self.target_column}' not found in CSV")
            
            # Select feature columns
            if self.feature_columns is None:
                feature_columns = [col for col in data.columns if col != self.target_column]
            else:
                feature_columns = self.feature_columns
                # Validate feature columns exist
                missing_cols = [col for col in feature_columns if col not in data.columns]
                if missing_cols:
                    raise DataLoadingError(f"Feature columns not found: {missing_cols}")
            
            # Select features and target
            features = data[feature_columns].copy()
            target = data[self.target_column].copy()
            
            # Apply preprocessing steps
            if self.preprocessing.get('impute_missing', False):
                features = self._impute_missing_values(features)
            
            if self.preprocessing.get('standardize', False):
                features = self._standardize_features(features)
            
            if self.preprocessing.get('encode_categoricals', False):
                features = self._encode_categorical_features(features)
            
            # Apply label mapping if provided
            if self.label_mapping:
                target = target.map(self.label_mapping)
                # Check for unmapped values
                if target.isna().any():
                    unmapped_values = data[self.target_column][target.isna()].unique()
                    self.logger.warning(f"Unmapped label values found: {unmapped_values}")
                    # Drop rows with unmapped labels
                    mapped_mask = ~target.isna()
                    features = features[mapped_mask]
                    target = target[mapped_mask]
            
            # Combine features and target
            processed_data = pd.concat([features, target], axis=1)
            
            # Remove any remaining NaN rows
            initial_size = len(processed_data)
            processed_data = processed_data.dropna()
            final_size = len(processed_data)
            
            if final_size < initial_size:
                self.logger.info(f"Removed {initial_size - final_size} rows with NaN values")
            
            if len(processed_data) == 0:
                raise DataLoadingError("No valid samples remaining after preprocessing")
            
            return processed_data
            
        except Exception as e:
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(f"Data preprocessing failed: {e}") from e
    
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
            else:  # constant
                fill_value = self.preprocessing.get('fill_value', 0)
                features[numeric_cols] = features[numeric_cols].fillna(fill_value)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_values = features[col].mode()
                if len(mode_values) > 0:
                    features[col] = features[col].fillna(mode_values.iloc[0])
                else:
                    features[col] = features[col].fillna('unknown')
        
        return features
    
    def _standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize numeric features."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            means = features[numeric_cols].mean()
            stds = features[numeric_cols].std()
            
            # Avoid division by zero
            stds = stds.replace(0, 1)
            
            features[numeric_cols] = (features[numeric_cols] - means) / stds
        
        return features
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns
        
        for col in categorical_cols:
            # Create mapping for consistent encoding
            unique_values = sorted(features[col].unique())
            value_to_int = {val: idx for idx, val in enumerate(unique_values)}
            features[col] = features[col].map(value_to_int)
        
        return features
    
    def _convert_to_tensors(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pandas DataFrame to PyTorch tensors.
        
        Args:
            data: Processed pandas DataFrame
            
        Returns:
            Tuple of (features_tensor, targets_tensor)
        """
        # Separate features and targets
        feature_columns = [col for col in data.columns if col != self.target_column]
        features = data[feature_columns].values.astype(np.float32)
        targets = data[self.target_column].values
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        
        # Handle different target types
        if targets.dtype.kind in ['i', 'u']:  # Integer types
            targets_tensor = torch.from_numpy(targets).long()
        elif targets.dtype.kind == 'f':  # Float types
            targets_tensor = torch.from_numpy(targets.astype(np.float32))
        else:  # Convert to int64 for classification
            targets_tensor = torch.from_numpy(targets.astype(np.int64))
        
        return features_tensor, targets_tensor


class MNISTDataLoader(BaseDataLoader):
    """
    Production-grade MNIST data loader implementing DataLoaderInterface.
    
    Features:
    - Configurable binary classification
    - Automatic download and caching
    - Proper train/test splits
    - Standardized preprocessing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MNIST data loader.
        
        Args:
            config: Configuration dictionary containing:
                - path: Data directory path
                - binary_classification: Binary classification settings
                - transform_config: Transform configuration
        """
        super().__init__(config)
        
        self.data_path = self.config.get('path', './data/MNIST')
        self.binary_config = self.config.get('binary_classification')
        self.transform_config = self.config.get('transform_config', {})
        
        self.logger.info(f"Initialized MNIST data loader with path: {self.data_path}")
    
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load MNIST dataset.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Raises:
            DataLoadingError: If data loading fails
        """
        try:
            self.logger.info("Loading MNIST dataset...")
            
            # Create transforms
            transform = self._create_transforms()
            
            # Load MNIST datasets
            train_dataset = datasets.MNIST(
                root=self.data_path,
                train=True,
                download=True,
                transform=transform
            )
            
            test_dataset = datasets.MNIST(
                root=self.data_path,
                train=False,
                download=True,
                transform=transform
            )
            
            # Apply binary classification if configured
            if self.binary_config and self.binary_config.get('enabled', False):
                self.logger.info("Applying binary classification conversion...")
                train_dataset = self._convert_to_binary(train_dataset)
                test_dataset = self._convert_to_binary(test_dataset)
            
            self.logger.info(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return train_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load MNIST data: {e}")
            raise DataLoadingError(f"MNIST data loading failed: {e}") from e
    
    def get_data_info(self) -> DatasetInfo:
        """
        Get information about the MNIST dataset.
        
        Returns:
            DatasetInfo with dataset characteristics
        """
        # Standard MNIST info
        num_classes = 2 if (self.binary_config and self.binary_config.get('enabled', False)) else 10
        
        return {
            'name': self.config['name'],
            'size': 60000,  # Standard MNIST training size
            'num_features': 784,  # 28x28 flattened
            'num_classes': num_classes,
            'data_shape': (28, 28),
            'channels': 1,
            'type': 'image'
        }
    
    def _create_transforms(self) -> transforms.Compose:
        """Create transform pipeline."""
        transform_list = [transforms.ToTensor()]
        
        # Add normalization
        mean = self.transform_config.get('mean', 0.1307)
        std = self.transform_config.get('std', 0.3081)
        transform_list.append(transforms.Normalize((mean,), (std,)))
        
        # Add any additional transforms
        if self.transform_config.get('flatten', False):
            transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        
        return transforms.Compose(transform_list)
    
    def _convert_to_binary(self, dataset: Dataset) -> Dataset:
        """Convert multi-class dataset to binary classification."""
        normal_classes = self.binary_config.get('normal_classes', [0, 1, 2, 3, 4, 5, 6, 8, 9])
        anomaly_classes = self.binary_config.get('anomaly_classes', [7])
        
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


class DataLoaderRegistry:
    """
    Registry for data loaders following the Registry pattern.
    
    Allows easy extension and configuration-driven data loader selection.
    """
    
    _loaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, loader_class: type) -> None:
        """Register a data loader class."""
        if not issubclass(loader_class, DataLoaderInterface):
            raise ConfigurationError(f"Loader class must implement DataLoaderInterface")
        
        cls._loaders[name] = loader_class
        logger.info(f"Registered data loader: {name}")
    
    @classmethod
    def get_loader(cls, name: str) -> type:
        """Get a data loader class by name."""
        if name not in cls._loaders:
            raise ConfigurationError(f"Unknown data loader: {name}")
        
        return cls._loaders[name]
    
    @classmethod
    def create_loader(cls, config: Dict[str, Any]) -> DataLoaderInterface:
        """Create a data loader instance from configuration."""
        loader_name = config.get('name')
        if not loader_name:
            raise ConfigurationError("Data loader name not specified in config")
        
        loader_class = cls.get_loader(loader_name)
        return loader_class(config)
    
    @classmethod
    def list_loaders(cls) -> List[str]:
        """List all registered data loaders."""
        return list(cls._loaders.keys())


# Register built-in data loaders
DataLoaderRegistry.register('csv', CSVDataLoader)
DataLoaderRegistry.register('mnist', MNISTDataLoader)
