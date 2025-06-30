"""
data_loader.py

Provides configuration management, JSON-based logging, data preprocessing,
splitting and PyTorch DataLoader construction for a deep learning workflow.
Includes MNIST dataset support for federated learning experiments.
"""

import os, yaml, json, logging
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, List, Dict, Optional, Union
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class ConfigManager:
    """
    Handles loading and retrieval of hierarchical configuration values from YAML.

    Reads a YAML file at initialization ('config.yaml') and provides
    nested key lookup with dot notation.

    Attributes:
        cfg (dict): Loaded configuration dictionary.
    """
    def __init__(self, path="config.yaml"):
        """
        Args:
            path: Filesystem path to the YAML configuration file.
        """
        p = Path(path)
        self.cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot-separated keys.

        Args:
            key: Dot-notation path (e.g. "data.target_column").
            default: Value to return if key is missing.

        Returns:
            The configuration value, or default if not found.
        """
        v = self.cfg
        for k in key.split("."):
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return default
        return v

class JsonLogger:
    """
    A simple JSON-format logger that writes timestamped entries to a file.

    Attributes:
        l (logging.Logger): Underlying Python logger instance.
    """
    def __init__(self, file: str, level: str = "INFO"):
        """
        Args:
            file: Path to the log file. Parent directories will be created.
            level: Logging level as a string (e.g. "DEBUG", "INFO").
        """
        p = Path(file); p.parent.mkdir(parents=True, exist_ok=True)
        self.l = logging.getLogger("dl"); self.l.setLevel(getattr(logging, level))
        h = logging.FileHandler(file, "w"); h.setFormatter(logging.Formatter("%(message)s"))
        self.l.handlers = [h]
    def log(self, level: str, msg: str, **kw):
        """
        Write a JSON-encoded log entry with timestamp, level and message.

        Args:
            level: Severity level (e.g. "INFO", "ERROR").
            msg: Human-readable message.
            **kw: Additional key-value pairs to include in the entry.
        """
        entry = {"timestamp": datetime.now().isoformat(), "level": level, "message": msg, **kw}
        getattr(self.l, level.lower())(json.dumps(entry))

class DataPreprocessor:
    """
    Encapsulates feature encoding, imputation, scaling and label encoding.

    On fit, determines numeric and categorical columns, fits transformers
    (SimpleImputer + StandardScaler) for numeric data and LabelEncoder
    for each categorical feature.

    Attributes:
        cm: ConfigManager instance for retrieving settings.
        lg: JsonLogger instance for structured logging.
        le (dict[str, LabelEncoder]): Fitted encoders for categorical columns.
        ct (ColumnTransformer): Combined numeric transformer pipeline.
    """
    def __init__(self, cm: ConfigManager, lg: JsonLogger):
        """
        Args:
            cm: ConfigManager instance.
            lg: JsonLogger instance.
        """
        self.cm, self.lg, self.le, self.ct = cm, lg, {}, None
    def fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit transformers on the DataFrame and transform features.

        Args:
            df: Pandas DataFrame containing features and target.

        Returns:
            A tuple (X_transformed, y_array), where X_transformed is a numpy
            array of transformed features, and y_array is the target values.
        """
        tc = self.cm.get("data.target_column")
        y = df[tc].values if tc and tc in df else None
        X = df.drop(columns=[tc]) if tc and tc in df else df.copy()
        num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        cat = [c for c in X if c not in num]
        for c in cat:
            le = LabelEncoder().fit(X[c].astype(str)); self.le[c] = le
            X[c] = le.transform(X[c].astype(str))
        if num:
            self.ct = ColumnTransformer([
                ("n", Pipeline([("im", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num)
            ], remainder="drop")
            Xt = self.ct.fit_transform(X)
        else:
            Xt = X.values
        return Xt, y
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted transformations to new data.

        Args:
            df: New DataFrame with same schema as used in fit().

        Returns:
            Numpy array of transformed features.
        """
        tc = self.cm.get("data.target_column")
        X = df.drop(columns=[tc]) if tc and tc in df else df.copy()
        for c, le in self.le.items():
            X[c] = le.transform(X[c].astype(str))
        return self.ct.transform(X) if self.ct else X.values

class ClientDataset(Dataset):
    """
    PyTorch Dataset wrapping feature and label arrays.

    Converts numpy arrays to torch tensors of appropriate dtype.

    Attributes:
        X (torch.Tensor): Features of shape (N, D), dtype float32.
        y (torch.Tensor): Integer class labels of shape (N,), dtype long.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: 2D numpy array of input features.
            y: 1D numpy array of target labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        """Return the number of samples."""
        return len(self.X)
    def __getitem__(self, i): 
        """
        Retrieve one sample by index.

        Args:
            idx: Index of the sample.

        Returns:
            A tuple (feature_tensor, label_tensor).
        """
        return self.X[i], self.y[i]

class DataSplitter:
    """
    Splits arrays into train, test and validation sets according to configuration.

    Uses sklearn.model_selection.train_test_split with optional stratification.

    Attributes:
        cm: ConfigManager instance for retrieving split ratios.
        lg: JsonLogger instance for recording split sizes.
    """
    def __init__(self, cm: ConfigManager, lg: JsonLogger):
        """
        Args:
            cm: Configuration manager.
            lg: Logger for recording the split operation.
        """
        self.cm, self.lg = cm, lg
    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Perform train/test/validation split.

        Reads ratios 'split.train_ratio', 'split.test_ratio' and 'split.val_ratio',
        ensures they sum to 1, and applies two-stage splitting. Logs counts.

        Args:
            X: Feature array.
            y: Label array.

        Returns:
            A tuple of three (X_sub, y_sub) pairs:
            (train, test, validation).
        """
        tr, te, va = (self.cm.get(k, d) for k, d in [
            ("split.train_ratio", 0.8),
            ("split.test_ratio", 0.15),
            ("split.val_ratio", 0.05),
        ])
        rs = self.cm.get("split.random_state", 42) # random_state
        st = self.cm.get("split.stratify", True) and y is not None # stratify_flag
        assert abs(tr + te + va - 1) < 1e-6 # Ensure ratios sum to 1
        tv = te + va

        # First split: train vs. temp (test+val)
        xtr, xtv, ytr, ytv = train_test_split(X, y, test_size=tv,
                                              stratify=y if st else None, random_state=rs)
        v = va / tv
        # Second split: test vs. val
        xte, xva, yte, yva = train_test_split(xtv, ytv, test_size=v,
                                              stratify=ytv if st else None, random_state=rs)
        self.lg.log("INFO", "split", train=len(xtr), test=len(xte), val=len(xva))
        return (xtr, ytr), (xte, yte), (xva, yva)

class DataLoaderFactory:
    """
    Orchestrates end-to-end data loading: reading CSV, preprocessing, splitting,
    and wrapping in PyTorch DataLoaders.

    Configuration options:
        data.file_path, data.target_column,
        logging.log_file, logging.log_level,
        split.* ratios and random_state.

    Attributes:
        cm: ConfigManager instance.
        lg: JsonLogger instance.
        dp: DataPreprocessor instance.
        sp: DataSplitter instance.
    """
    def __init__(self, cfg: str = "config.yaml"):
        """
        Args:
            cfg: Path to configuration YAML.
        """
        self.cm = ConfigManager(cfg)
        self.lg = JsonLogger(self.cm.get("logging.log_file", "logs/data_loader.json"),
                             self.cm.get("logging.log_level", "INFO"))
        self.dp = DataPreprocessor(self.cm, self.lg)
        self.sp = DataSplitter(self.cm, self.lg)
        self.lg.log("INFO", "init")
    def load(self) -> pd.DataFrame:
        """
        Read the raw data CSV into a DataFrame.

        Returns:
            Pandas DataFrame of the entire dataset.
        """
        fp = self.cm.get("data.file_path", "data/clients.csv")
        df = pd.read_csv(fp)
        self.lg.log("INFO", "loaded", shape=df.shape)
        return df
    def create(self, bs: int = 32, nw: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build DataLoaders for training, validation and testing.

        Args:
            bs: Batch size for all DataLoaders.
            nw: Number of worker processes for loading.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        df = self.load()
        tc = self.cm.get("data.target_column")
        df = df.dropna(subset=[tc]) if tc and tc in df else df
        Xt, y = self.dp.fit(df)
        (xtr, ytr), (xte, yte), (xva, yva) = self.sp.split(Xt, y)
        ds = lambda X, y: ClientDataset(X, y)
        loaders = [
            DataLoader(ds(xtr, ytr), batch_size=bs, shuffle=True, num_workers=nw),
            DataLoader(ds(xva, yva), batch_size=bs, shuffle=False, num_workers=nw),
            DataLoader(ds(xte, yte), batch_size=bs, shuffle=False, num_workers=nw),
        ]
        self.lg.log("INFO", "loaders", batch_size=bs, workers=nw)
        return tuple(loaders)

class MNISTDatasetWrapper(Dataset):
    """
    Wrapper for MNIST dataset to support anomaly detection experiments.
    Treats specific digits as anomalies for binary classification.
    """
    
    def __init__(self, mnist_dataset: Dataset, anomaly_digits: List[int] = [1, 7], transform=None):
        """
        Initialize MNIST wrapper for anomaly detection.
        
        Args:
            mnist_dataset: Original MNIST dataset
            anomaly_digits: List of digits to treat as anomalies (default: [1, 7])
            transform: Additional transforms to apply
        """
        self.mnist_dataset = mnist_dataset
        self.anomaly_digits = set(anomaly_digits)
        self.transform = transform
        
        # Convert to binary labels (0: normal, 1: anomaly)
        self.binary_labels = []
        for i in range(len(mnist_dataset)):
            _, label = mnist_dataset[i]
            self.binary_labels.append(1 if label in self.anomaly_digits else 0)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created MNIST anomaly dataset. Anomaly digits: {anomaly_digits}")
        self.logger.info(f"Normal samples: {self.binary_labels.count(0)}, "
                        f"Anomaly samples: {self.binary_labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.mnist_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Flatten image for MLP
        image = image.view(-1)  # 28*28 = 784 features
        
        return image, torch.tensor(self.binary_labels[idx], dtype=torch.long)


class MNISTDataLoader:
    """
    MNIST dataset loader for federated learning with anomaly detection.
    Provides data splitting, preprocessing, and client distribution.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize MNIST data loader.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.cm = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Setup data directory
        self.data_dir = Path(self.cm.get("data.data_dir", "data/MNIST"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # MNIST specific configuration
        self.anomaly_digits = self.cm.get("data.anomaly_digits", [1, 7])
        self.download = self.cm.get("data.download", True)
        
        # Data splitting configuration
        self.test_size = self.cm.get("data.test_size", 0.2)
        self.val_size = self.cm.get("data.validation_size", 0.1)
        self.random_state = self.cm.get("data.random_state", 42)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Initialize datasets
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        
    def load_mnist_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load MNIST train and test datasets with error handling.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        try:
            self.logger.info(f"Loading MNIST data from {self.data_dir}")
            
            # Load MNIST datasets
            mnist_train = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=self.download,
                transform=self.transform
            )
            
            mnist_test = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=self.download,
                transform=self.transform
            )
            
            # Wrap for anomaly detection
            train_dataset = MNISTDatasetWrapper(mnist_train, self.anomaly_digits)
            test_dataset = MNISTDatasetWrapper(mnist_test, self.anomaly_digits)
            
            self.logger.info(f"Successfully loaded MNIST data. "
                           f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
            return train_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load MNIST data: {e}")
            raise RuntimeError(f"MNIST data loading failed: {e}")
    
    def create_validation_split(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Create validation split from training data.
        
        Args:
            train_dataset: Training dataset to split
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        try:
            train_size = int((1 - self.val_size) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            # Set random seed for reproducibility
            torch.manual_seed(self.random_state)
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size]
            )
            
            self.logger.info(f"Created validation split. "
                           f"Train: {len(train_subset)}, Val: {len(val_subset)}")
            
            return train_subset, val_subset
            
        except Exception as e:
            self.logger.error(f"Failed to create validation split: {e}")
            raise
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load MNIST data
        full_train, test_dataset = self.load_mnist_data()
        
        # Create validation split
        train_dataset, val_dataset = self.create_validation_split(full_train)
        
        # Store for later use
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Log dataset statistics
        self._log_dataset_statistics()
        
        return train_dataset, val_dataset, test_dataset
    
    def _log_dataset_statistics(self):
        """Log comprehensive dataset statistics."""
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            return
        
        stats = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "anomaly_digits": self.anomaly_digits,
            "total_samples": len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        }
        
        # Count anomalies in each split
        for split_name, dataset in [("train", self.train_dataset), 
                                   ("val", self.val_dataset), 
                                   ("test", self.test_dataset)]:
            normal_count = 0
            anomaly_count = 0
            
            # Sample a subset to count labels efficiently
            sample_size = min(1000, len(dataset))
            indices = torch.randperm(len(dataset))[:sample_size]
            
            for idx in indices:
                _, label = dataset[idx]
                if label.item() == 0:
                    normal_count += 1
                else:
                    anomaly_count += 1
            
            # Estimate full split statistics
            total_size = len(dataset)
            estimated_normal = int((normal_count / sample_size) * total_size)
            estimated_anomaly = total_size - estimated_normal
            
            stats[f"{split_name}_normal_estimated"] = estimated_normal
            stats[f"{split_name}_anomaly_estimated"] = estimated_anomaly
        
        self.logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        return stats
    
    def create_federated_datasets(
        self, 
        num_clients: int, 
        distribution: str = "iid",
        alpha: float = 0.5
    ) -> List[Tuple[Dataset, Dataset]]:
        """
        Create federated datasets for multiple clients.
        
        Args:
            num_clients: Number of federated clients
            distribution: "iid" or "non_iid"
            alpha: Dirichlet parameter for non-IID distribution
            
        Returns:
            List of (train_dataset, test_dataset) tuples for each client
        """
        try:
            if not self.train_dataset:
                self.prepare_datasets()
            
            self.logger.info(f"Creating federated datasets for {num_clients} clients "
                           f"with {distribution} distribution")
            
            if distribution == "iid":
                return self._create_iid_split(num_clients)
            elif distribution == "non_iid":
                return self._create_non_iid_split(num_clients, alpha)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
                
        except Exception as e:
            self.logger.error(f"Failed to create federated datasets: {e}")
            raise
    
    def _create_iid_split(self, num_clients: int) -> List[Tuple[Dataset, Dataset]]:
        """Create IID split of datasets among clients."""
        # Split training data
        train_size = len(self.train_dataset)
        client_train_sizes = [train_size // num_clients] * num_clients
        
        # Distribute remainder
        for i in range(train_size % num_clients):
            client_train_sizes[i] += 1
        
        # Random split training data
        torch.manual_seed(self.random_state)
        train_splits = random_split(self.train_dataset, client_train_sizes)
        
        # Split test data similarly
        test_size = len(self.test_dataset)
        client_test_sizes = [test_size // num_clients] * num_clients
        
        for i in range(test_size % num_clients):
            client_test_sizes[i] += 1
        
        torch.manual_seed(self.random_state + 1)
        test_splits = random_split(self.test_dataset, client_test_sizes)
        
        client_datasets = list(zip(train_splits, test_splits))
        
        self.logger.info(f"Created IID split: {[len(train) for train, _ in client_datasets]} "
                        f"train samples per client")
        
        return client_datasets
    
    def _create_non_iid_split(self, num_clients: int, alpha: float) -> List[Tuple[Dataset, Dataset]]:
        """Create non-IID split using Dirichlet distribution."""
        # This is a simplified non-IID split
        # In practice, you might want to implement a more sophisticated approach
        
        # Get labels for sorting
        train_indices = list(range(len(self.train_dataset)))
        
        # Sort by labels to create label skew
        def get_label(idx):
            _, label = self.train_dataset[idx]
            return label.item()
        
        train_indices.sort(key=get_label)
        
        # Split indices among clients with some overlap
        client_datasets = []
        indices_per_client = len(train_indices) // num_clients
        
        for i in range(num_clients):
            start_idx = i * indices_per_client
            end_idx = start_idx + indices_per_client
            if i == num_clients - 1:  # Last client gets remaining
                end_idx = len(train_indices)
            
            client_train_indices = train_indices[start_idx:end_idx]
            
            # Create subset for this client
            client_train = Subset(self.train_dataset, client_train_indices)
            
            # For test data, give each client a random subset
            test_indices = torch.randperm(len(self.test_dataset))[:len(self.test_dataset) // num_clients]
            client_test = Subset(self.test_dataset, test_indices.tolist())
            
            client_datasets.append((client_train, client_test))
        
        self.logger.info(f"Created non-IID split: {[len(train) for train, _ in client_datasets]} "
                        f"train samples per client")
        
        return client_datasets
    
    def get_centralized_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get centralized DataLoaders for baseline comparison.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if not self.train_dataset:
            self.prepare_datasets()
        
        batch_size = self.cm.get("client.batch_size", 32)
        num_workers = self.cm.get("data.num_workers", 0)  # 0 for IoT devices
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        self.logger.info(f"Created centralized DataLoaders with batch_size={batch_size}")
        
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    """
    Example usage:
        $ python data_pipeline.py
    Prints the number of batches in each DataLoader.
    """
    tl, vl, el = DataLoaderFactory().create(64)
    print([len(l) for l in (tl, vl, el)])
