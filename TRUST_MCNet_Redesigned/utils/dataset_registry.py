"""
Dataset registry for federated learning.

This module implements a registry pattern for dataset loading,
supporting various datasets and avoiding if/else chains.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Load dataset based on configuration.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (train_dataset, test_dataset)
            test_dataset can be None if not available
        """
        pass
    
    @abstractmethod
    def get_data_shape(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """
        Get the shape of input data.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Shape tuple (channels, height, width) or (features,)
        """
        pass
    
    @abstractmethod
    def get_num_classes(self, config: Dict[str, Any]) -> int:
        """
        Get number of classes in the dataset.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Number of classes
        """
        pass


class MNISTLoader(DatasetLoader):
    """MNIST dataset loader."""
    
    def load(self, config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
        """Load MNIST dataset."""
        data_path = config['path']
        
        # Create transforms
        transform_list = [transforms.ToTensor()]
        
        if config.get('transforms', {}).get('normalize', False):
            mean = config['transforms'].get('mean', [0.1307])
            std = config['transforms'].get('std', [0.3081])
            transform_list.append(transforms.Normalize(mean, std))
        
        transform = transforms.Compose(transform_list)
        
        try:
            # Load train and test datasets
            train_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            
            test_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )
            
            # Handle binary classification if configured
            binary_config = config.get('binary_classification')
            if binary_config and binary_config.get('enabled', False):
                train_dataset = self._create_binary_dataset(train_dataset, binary_config)
                test_dataset = self._create_binary_dataset(test_dataset, binary_config)
            
            logger.info(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            raise
    
    def get_data_shape(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Get MNIST data shape."""
        return (1, 28, 28)  # 1 channel, 28x28 pixels
    
    def get_num_classes(self, config: Dict[str, Any]) -> int:
        """Get number of MNIST classes."""
        binary_config = config.get('binary_classification')
        if binary_config and binary_config.get('enabled', False):
            return 2  # Binary classification
        return 10  # Standard MNIST
    
    def _create_binary_dataset(self, dataset: Dataset, binary_config: Dict[str, Any]) -> Dataset:
        """Create binary classification dataset from MNIST."""
        normal_classes = binary_config.get('normal_classes', [0, 1, 2, 3, 4, 5, 6, 8, 9])
        anomaly_classes = binary_config.get('anomaly_classes', [7])
        
        return BinaryMNIST(dataset, normal_classes, anomaly_classes)


class CIFAR10Loader(DatasetLoader):
    """CIFAR-10 dataset loader."""
    
    def load(self, config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
        """Load CIFAR-10 dataset."""
        data_path = config['path']
        
        # Create transforms
        transform_list = [transforms.ToTensor()]
        
        if config.get('transforms', {}).get('normalize', False):
            mean = config['transforms'].get('mean', [0.485, 0.456, 0.406])
            std = config['transforms'].get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean, std))
        
        transform = transforms.Compose(transform_list)
        
        try:
            # Load train and test datasets
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )
            
            logger.info(f"Loaded CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 dataset: {e}")
            raise
    
    def get_data_shape(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Get CIFAR-10 data shape."""
        return (3, 32, 32)  # 3 channels, 32x32 pixels
    
    def get_num_classes(self, config: Dict[str, Any]) -> int:
        """Get number of CIFAR-10 classes."""
        return 10


class CSVLoader(DatasetLoader):
    """CSV dataset loader."""
    
    def load(self, config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
        """Load CSV dataset."""
        csv_path = config['path']
        
        try:
            dataset = CSVDataset(
                csv_path=csv_path,
                target_column=config['csv']['target_column'],
                feature_columns=config['csv'].get('feature_columns'),
                preprocessing=config.get('preprocessing', {})
            )
            
            logger.info(f"Loaded CSV dataset: {len(dataset)} samples")
            return dataset, None  # No separate test set for CSV
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset: {e}")
            raise
    
    def get_data_shape(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Get CSV data shape."""
        # This needs to be determined from the actual data
        # For now, return a placeholder that will be updated after loading
        return (config.get('input_dim', 784),)
    
    def get_num_classes(self, config: Dict[str, Any]) -> int:
        """Get number of CSV classes."""
        return config.get('num_classes', 2)


class BinaryMNIST(Dataset):
    """Binary classification wrapper for MNIST dataset."""
    
    def __init__(self, original_dataset: Dataset, normal_classes: List[int], anomaly_classes: List[int]):
        """
        Initialize binary MNIST dataset.
        
        Args:
            original_dataset: Original MNIST dataset
            normal_classes: List of classes to label as normal (0)
            anomaly_classes: List of classes to label as anomaly (1)
        """
        self.original_dataset = original_dataset
        self.normal_classes = set(normal_classes)
        self.anomaly_classes = set(anomaly_classes)
        
        # Filter dataset to only include relevant classes
        self.indices = []
        for i in range(len(original_dataset)):
            _, label = original_dataset[i]
            if label in self.normal_classes or label in self.anomaly_classes:
                self.indices.append(i)
        
        logger.info(f"Binary MNIST: {len(self.indices)} samples "
                   f"(normal: {normal_classes}, anomaly: {anomaly_classes})")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        image, label = self.original_dataset[original_idx]
        
        # Convert to binary label (0: normal, 1: anomaly)
        binary_label = 1 if label in self.anomaly_classes else 0
        
        return image, binary_label


class CSVDataset(Dataset):
    """PyTorch Dataset for CSV data with preprocessing capabilities."""
    
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
                features = features.fillna(features.mean())
            
            if self.preprocessing.get('standardize', False):
                features = (features - features.mean()) / features.std()
            
            # Combine features and target
            processed_data = pd.concat([features, target], axis=1)
            
            # Remove any remaining NaN rows
            processed_data = processed_data.dropna()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to preprocess CSV data: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Features (all columns except target)
        features = row.drop(self.target_column).values.astype(np.float32)
        features = torch.tensor(features)
        
        # Target
        target = torch.tensor(row[self.target_column], dtype=torch.long)
        
        return features, target


class DatasetRegistry:
    """Registry for dataset loaders."""
    
    _loaders = {
        'mnist': MNISTLoader,
        'cifar10': CIFAR10Loader,
        'custom_csv': CSVLoader
    }
    
    @classmethod
    def get_loader(cls, dataset_name: str) -> DatasetLoader:
        """
        Get dataset loader by name.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dataset loader instance
            
        Raises:
            ValueError: If dataset is not registered
        """
        if dataset_name not in cls._loaders:
            available_datasets = list(cls._loaders.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available datasets: {available_datasets}")
        
        return cls._loaders[dataset_name]()
    
    @classmethod
    def register_loader(cls, name: str, loader_class: type) -> None:
        """
        Register a new dataset loader.
        
        Args:
            name: Dataset name
            loader_class: Loader class
        """
        if not issubclass(loader_class, DatasetLoader):
            raise ValueError("Loader class must inherit from DatasetLoader")
        
        cls._loaders[name] = loader_class
        logger.info(f"Registered dataset loader: {name}")
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available datasets."""
        return list(cls._loaders.keys())


class DataManager:
    """Centralized data management for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data manager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.loader = DatasetRegistry.get_loader(config['name'])
        self.train_dataset = None
        self.test_dataset = None
        
    def load_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Load train and test datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        try:
            self.train_dataset, self.test_dataset = self.loader.load(self.config)
            logger.info(f"Successfully loaded {self.config['name']} dataset")
            return self.train_dataset, self.test_dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {self.config['name']}: {e}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'data_shape': self.loader.get_data_shape(self.config),
            'num_classes': self.loader.get_num_classes(self.config),
            'dataset_name': self.config['name']
        }
