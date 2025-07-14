"""
MedBIoT Dataset implementation for TRUST_MCNet.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

from . import register, get_data_root


@register("medbiot")
class MedBIoTDataset:
    """
    MedBIoT dataset loader for federated learning.
    
    Provides standardized interface for loading MedBIoT medical IoT data
    with train/test splits and preprocessing.
    """
    
    def __init__(self, batch_size: int = 32, data_root: Optional[str] = None, 
                 test_split: float = 0.2, normalize: bool = True):
        """
        Initialize MedBIoT dataset.
        
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
        """Load and preprocess MedBIoT data."""
        data_path = os.path.join(self.data_root, "IoT_Datasets", "MedBIoT")
        
        # Check if data exists, create synthetic data if not
        if not os.path.exists(data_path):
            print(f"Warning: MedBIoT data not found at {data_path}. Creating synthetic data.")
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
            print(f"Error loading MedBIoT data: {e}. Creating synthetic data.")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic medical IoT device data for testing."""
        np.random.seed(44)  # Different seed for variety
        
        # Generate synthetic medical device features
        n_samples = 3500
        n_features = 115  # Medical IoT datasets often have many features
        
        # Normal medical device operations (85%)
        normal_samples = int(0.85 * n_samples)
        normal_data = np.random.normal(0, 0.5, (normal_samples, n_features))
        
        # Blackhole attacks (5%)
        blackhole_samples = int(0.05 * n_samples)
        blackhole_data = np.random.normal(-3, 1, (blackhole_samples, n_features))
        
        # Flooding attacks (5%)
        flooding_samples = int(0.05 * n_samples)
        flooding_data = np.random.normal(4, 2, (flooding_samples, n_features))
        
        # MQTT Publish attacks (3%)
        mqtt_samples = int(0.03 * n_samples)
        mqtt_data = np.random.normal(2, 1.5, (mqtt_samples, n_features))
        
        # Other attacks (2%)
        other_samples = n_samples - normal_samples - blackhole_samples - flooding_samples - mqtt_samples
        other_data = np.random.normal(-1, 2, (other_samples, n_features))
        
        # Combine data
        X = np.vstack([normal_data, blackhole_data, flooding_data, mqtt_data, other_data])
        y = np.hstack([
            np.zeros(normal_samples),      # Normal: 0
            np.ones(blackhole_samples),    # Blackhole: 1
            np.full(flooding_samples, 2),  # Flooding: 2
            np.full(mqtt_samples, 3),      # MQTT: 3
            np.full(other_samples, 4)      # Other: 4
        ])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        self._input_dim = n_features
        self._num_classes = 5
        
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
            "name": "MedBIoT",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "batch_size": self.batch_size
        }
