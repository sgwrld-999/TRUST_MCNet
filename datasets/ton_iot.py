"""
ToN-IoT Dataset implementation for TRUST_MCNet.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

from . import register, get_data_root


@register("ton_iot")
class ToNIoTDataset:
    """
    ToN-IoT dataset loader for federated learning.
    
    Provides standardized interface for loading ToN-IoT network traffic data
    with train/test splits and preprocessing.
    """
    
    def __init__(self, batch_size: int = 32, data_root: Optional[str] = None, 
                 test_split: float = 0.2, normalize: bool = True):
        """
        Initialize ToN-IoT dataset.
        
        Args:
            batch_size: Batch size for data loaders
            data_root: Root directory for data files
            test_split: Fraction of data for testing
            normalize: Whether to normalize features
        """
        self.batch_size = batch_size
        self.data_root = data_root or get_data_root()
        self.test_split = test_split
        self.normalize = normalize
        
        self._input_dim = None
        self._num_classes = None
        self.train_dataset = None
        self.test_dataset = None
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess ToN-IoT data."""
        data_path = os.path.join(self.data_root, "IoT_Datasets", "ToN_IoT")
        
        # Check if data exists, create synthetic data if not
        if not os.path.exists(data_path):
            print(f"Warning: ToN-IoT data not found at {data_path}. Creating synthetic data.")
            self._create_synthetic_data()
            return
        
        try:
            # Try to load actual data files
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if not csv_files:
                print("No CSV files found. Creating synthetic data.")
                self._create_synthetic_data()
                return
            
            # Load the first CSV file found
            df = pd.read_csv(os.path.join(data_path, csv_files[0]))
            self._process_dataframe(df)
            
        except Exception as e:
            print(f"Error loading ToN-IoT data: {e}. Creating synthetic data.")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic IoT network traffic data for testing."""
        np.random.seed(42)
        
        # Generate synthetic network features
        n_samples = 5000
        n_features = 42  # Common number of features in IoT datasets
        
        # Normal traffic (80%)
        normal_samples = int(0.8 * n_samples)
        normal_data = np.random.normal(0, 1, (normal_samples, n_features))
        normal_labels = np.zeros(normal_samples)
        
        # Anomalous traffic (20%)
        anomaly_samples = n_samples - normal_samples
        anomaly_data = np.random.normal(2, 1.5, (anomaly_samples, n_features))
        anomaly_labels = np.ones(anomaly_samples)
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([normal_labels, anomaly_labels])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        self._input_dim = n_features
        self._num_classes = 2
        
        self._create_train_test_split(X, y)
    
    def _process_dataframe(self, df: pd.DataFrame):
        """Process real dataframe."""
        # Assume last column is label
        feature_columns = df.columns[:-1]
        label_column = df.columns[-1]
        
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle categorical labels
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        self._input_dim = X.shape[1]
        self._num_classes = len(np.unique(y))
        
        self._create_train_test_split(X, y)
    
    def _create_train_test_split(self, X: np.ndarray, y: np.ndarray):
        """Create train/test split and convert to PyTorch datasets."""
        # Split data
        n_train = int((1 - self.test_split) * len(X))
        
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        # Normalize features if requested
        if self.normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    def train_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get training data loader.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Training DataLoader
        """
        batch_size = batch_size or self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get test data loader.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Test DataLoader
        """
        batch_size = batch_size or self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    @property
    def input_dim(self) -> int:
        """Get input feature dimension."""
        return self._input_dim
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self._num_classes
    
    def __len__(self) -> int:
        """Get total dataset size."""
        return len(self.train_dataset) + len(self.test_dataset)
    
    def get_info(self) -> dict:
        """Get dataset information."""
        return {
            "name": "ToN-IoT",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "batch_size": self.batch_size
        }
