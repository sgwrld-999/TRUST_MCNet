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
            
            # Update data shape after loading for loaders that support it
            if hasattr(self.loader, '_actual_data_shape'):
                self._actual_data_shape = self.loader._actual_data_shape
            
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
        # Use actual data shape if available
        if hasattr(self, '_actual_data_shape'):
            data_shape = self._actual_data_shape
        else:
            data_shape = self.loader.get_data_shape(self.config)
            
        return {
            'data_shape': data_shape,
            'num_classes': self.loader.get_num_classes(self.config),
            'dataset_name': self.config['name']
        }


class IoTGeneralLoader(DatasetLoader):
    """General IoT dataset loader for network traffic anomaly detection."""
    
    def load(self, config: Dict[str, Any]) -> Tuple[Dataset, Optional[Dataset]]:
        """Load IoT dataset from CSV files."""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        import glob
        import os
        
        data_path = config['path']
        logger.info(f"Loading IoT datasets from: {data_path}")
        
        try:
            # Auto-detect CSV files or use specified files
            dataset_files = config.get('dataset_files', [])
            if not dataset_files:
                csv_pattern = os.path.join(data_path, "*.csv")
                dataset_files = glob.glob(csv_pattern)
                logger.info(f"Auto-detected {len(dataset_files)} CSV files")
            else:
                dataset_files = [os.path.join(data_path, f) for f in dataset_files]
            
            if not dataset_files:
                raise ValueError(f"No CSV files found in {data_path}")
            
            # Load and combine all datasets with proper mixing
            all_dataframes = []
            dataset_labels = []  # Track which dataset each sample came from
            
            for idx, file_path in enumerate(dataset_files):
                logger.info(f"Loading dataset: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path)
                df['dataset_source'] = idx  # Add source tracking
                all_dataframes.append(df)
                dataset_labels.extend([idx] * len(df))
            
            # Combine all datasets
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined dataset shape: {combined_df.shape}")
            
            # CRITICAL FIX: Split BEFORE processing to ensure proper global test set
            # First split into train and global test
            test_size = config.get('eval_fraction', 0.2)
            
            # Check if stratification is possible
            label_counts = combined_df['Label'].value_counts()
            min_class_count = label_counts.min()
            
            # Only stratify if all classes have at least 2 samples
            if min_class_count >= 2:
                train_df, test_df = train_test_split(
                    combined_df, test_size=test_size, random_state=42, 
                    stratify=combined_df['Label']  # Stratify by label, not dataset source
                )
                logger.info(f"Used stratified split with min class count: {min_class_count}")
            else:
                logger.warning(f"Some classes have only {min_class_count} samples - using random split instead of stratified")
                train_df, test_df = train_test_split(
                    combined_df, test_size=test_size, random_state=42
                )
            
            # Remove dataset_source column after splitting
            train_df = train_df.drop('dataset_source', axis=1)
            test_df = test_df.drop('dataset_source', axis=1)
            
            # Now preprocess both splits with the same pipeline
            train_processed = self._preprocess_dataframe(train_df, config)
            test_processed = self._preprocess_dataframe(test_df, config)
            
            # Split features and labels for both
            X_train, y_train = self._prepare_features_labels(train_processed, config)
            X_test, y_test = self._prepare_features_labels(test_processed, config)
            
            # Create PyTorch datasets
            train_dataset = IoTDataset(X_train, y_train)
            test_dataset = IoTDataset(X_test, y_test)
            
            # Store actual data shape for model initialization
            self._actual_data_shape = (X_train.shape[1],)
            
            logger.info(f"Created IoT datasets: {len(train_dataset)} train, {len(test_dataset)} test")
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Failed to load IoT dataset: {e}")
            raise
    
    def _preprocess_dataframe(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced preprocessing with adaptive feature selection for federated learning."""
        preprocessing_config = config.get('preprocessing', {})
        
        logger.info(f"Starting enhanced preprocessing on dataset with shape: {df.shape}")
        
        # Step 1: Data crawling and analysis
        data_analysis = self._crawl_and_analyze_data(df, config)
        logger.info(f"Data analysis complete: {data_analysis['summary']}")
        
        # Step 2: Handle missing values first
        if preprocessing_config.get('handle_missing_values', True):
            strategy = preprocessing_config.get('missing_value_strategy', 'median')
            df = self._handle_missing_values(df, strategy)
            logger.info(f"After handling missing values: {df.shape}")
        
        # Step 3: Automatic redundant feature removal
        if preprocessing_config.get('auto_remove_redundant', True):
            df = self._remove_redundant_features(df, config, data_analysis)
            logger.info(f"After removing redundant features: {df.shape}")
        
        # Step 4: Convert non-numeric to numeric
        df = self._convert_categorical_to_numeric(df, config)
        logger.info(f"After categorical conversion: {df.shape}")
        
        # Step 5: Adaptive feature selection
        if preprocessing_config.get('adaptive_feature_selection', True):
            selected_features = self._adaptive_feature_selection(df, config, data_analysis)
            # Keep only selected features plus target column
            target_col = config.get('label_config', {}).get('target_column', 'Label')
            columns_to_keep = selected_features + [target_col] if target_col in df.columns else selected_features
            df = df[columns_to_keep]
            logger.info(f"After adaptive feature selection: {df.shape}")
        
        # Step 6: Manual exclusions (if specified)
        exclude_columns = preprocessing_config.get('exclude_columns', [])
        if exclude_columns:
            target_col = config.get('label_config', {}).get('target_column', 'Label')
            exclude_columns = [col for col in exclude_columns if col != target_col and col in df.columns]
            if exclude_columns:
                df = df.drop(columns=exclude_columns)
                logger.info(f"After manual exclusions: {df.shape}")
        
        logger.info(f"Final preprocessed dataset shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if strategy == 'median':
            # For numerical columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            # For categorical columns, use mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
        elif strategy == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        elif strategy == 'drop':
            df = df.dropna()
        
        return df
    
    def _crawl_and_analyze_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data analysis and crawling to understand dataset characteristics."""
        
        logger.info("Starting comprehensive data crawling and analysis...")
        
        analysis = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'column_types': {},
            'missing_values': {},
            'unique_value_ratios': {},
            'suspected_metadata': [],
            'suspected_ids': [],
            'suspected_timestamps': [],
            'suspected_addresses': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'constant_columns': [],
            'high_cardinality_columns': [],
            'correlation_groups': [],
            'data_patterns': {},
            'summary': ""
        }
        
        target_col = config.get('label_config', {}).get('target_column', 'Label')
        
        for col in df.columns:
            if col == target_col:
                continue
                
            # Basic type analysis
            dtype = df[col].dtype
            analysis['column_types'][col] = str(dtype)
            
            # Missing value analysis
            missing_ratio = df[col].isnull().sum() / len(df)
            analysis['missing_values'][col] = missing_ratio
            
            # Unique value analysis
            unique_ratio = df[col].nunique() / len(df)
            analysis['unique_value_ratios'][col] = unique_ratio
            
            # Sample values for pattern detection
            sample_values = df[col].dropna().head(20).astype(str)
            
            # Pattern detection
            col_lower = col.lower()
            
            # 1. Metadata detection
            metadata_patterns = ['id', 'uid', 'guid', 'uuid', 'index', 'idx']
            if any(pattern in col_lower for pattern in metadata_patterns):
                analysis['suspected_metadata'].append(col)
            
            # 2. ID detection (high cardinality + patterns)
            if unique_ratio > 0.8 and any(pattern in col_lower for pattern in ['id', 'uid', 'key']):
                analysis['suspected_ids'].append(col)
            
            # 3. Timestamp detection
            timestamp_patterns = ['time', 'date', 'timestamp', 'ts']
            if any(pattern in col_lower for pattern in timestamp_patterns):
                analysis['suspected_timestamps'].append(col)
            
            # Check for timestamp-like content
            if dtype == 'object' and len(sample_values) > 0:
                timestamp_like = sample_values.str.contains(
                    r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\d{10,13}', 
                    regex=True
                ).sum()
                if timestamp_like > len(sample_values) * 0.5:
                    analysis['suspected_timestamps'].append(col)
            
            # 4. Address detection (IP, MAC, host)
            address_patterns = ['host', 'ip', 'addr', 'mac', 'port']
            if any(pattern in col_lower for pattern in address_patterns):
                analysis['suspected_addresses'].append(col)
            
            # Check for IP-like content
            if dtype == 'object' and len(sample_values) > 0:
                ip_like = sample_values.str.contains(
                    r'\d+\.\d+\.\d+\.\d+|[0-9a-fA-F:]+', 
                    regex=True
                ).sum()
                if ip_like > len(sample_values) * 0.3:
                    analysis['suspected_addresses'].append(col)
            
            # 5. Categorize by data type
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                analysis['numeric_columns'].append(col)
                
                # Check if effectively constant
                if df[col].nunique() <= 2:
                    analysis['constant_columns'].append(col)
                    
            else:
                analysis['categorical_columns'].append(col)
                
            # 6. High cardinality detection
            if unique_ratio > 0.9 and len(df) > 50:
                analysis['high_cardinality_columns'].append(col)
            
            # 7. Store data patterns
            if len(sample_values) > 0:
                pattern_info = {
                    'sample_values': sample_values.tolist()[:5],
                    'avg_length': sample_values.str.len().mean() if dtype == 'object' else None,
                    'contains_numbers': sample_values.str.contains(r'\d', regex=True).any() if dtype == 'object' else None,
                    'contains_special_chars': sample_values.str.contains(r'[^a-zA-Z0-9\s]', regex=True).any() if dtype == 'object' else None
                }
                analysis['data_patterns'][col] = pattern_info
        
        # Remove duplicates from lists
        for key in ['suspected_metadata', 'suspected_ids', 'suspected_timestamps', 'suspected_addresses']:
            analysis[key] = list(set(analysis[key]))
        
        # Correlation analysis for numeric columns
        if len(analysis['numeric_columns']) > 1:
            try:
                numeric_df = df[analysis['numeric_columns']].fillna(df[analysis['numeric_columns']].median())
                corr_matrix = numeric_df.corr().abs()
                
                # Find highly correlated groups
                high_corr_threshold = 0.9
                corr_groups = []
                processed_cols = set()
                
                for i, col1 in enumerate(corr_matrix.columns):
                    if col1 in processed_cols:
                        continue
                    
                    corr_group = [col1]
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i != j and col2 not in processed_cols:
                            if corr_matrix.loc[col1, col2] > high_corr_threshold:
                                corr_group.append(col2)
                                processed_cols.add(col2)
                    
                    if len(corr_group) > 1:
                        corr_groups.append(corr_group)
                        processed_cols.update(corr_group)
                
                analysis['correlation_groups'] = corr_groups
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
        
        # Generate summary
        redundant_count = len(set(
            analysis['suspected_metadata'] + 
            analysis['suspected_ids'] + 
            analysis['suspected_timestamps'] + 
            analysis['suspected_addresses'] +
            analysis['constant_columns'] +
            analysis['high_cardinality_columns']
        ))
        
        analysis['summary'] = (
            f"Found {analysis['total_columns']} columns, "
            f"{len(analysis['numeric_columns'])} numeric, "
            f"{len(analysis['categorical_columns'])} categorical. "
            f"Identified {redundant_count} potentially redundant features."
        )
        
        logger.info(f"Data analysis results:")
        logger.info(f"  - Suspected metadata columns: {len(analysis['suspected_metadata'])}")
        logger.info(f"  - Suspected ID columns: {len(analysis['suspected_ids'])}")
        logger.info(f"  - Suspected timestamp columns: {len(analysis['suspected_timestamps'])}")
        logger.info(f"  - Suspected address columns: {len(analysis['suspected_addresses'])}")
        logger.info(f"  - Constant/near-constant columns: {len(analysis['constant_columns'])}")
        logger.info(f"  - High cardinality columns: {len(analysis['high_cardinality_columns'])}")
        
        return analysis
    
    def _remove_redundant_features(self, df: pd.DataFrame, config: Dict[str, Any], analysis: Dict[str, Any]) -> pd.DataFrame:
        """Remove clearly redundant features based on analysis."""
        
        target_col = config.get('label_config', {}).get('target_column', 'Label')
        original_cols = list(df.columns)
        
        # Features to remove
        to_remove = set()
        
        # Add suspected metadata and ID columns
        to_remove.update(analysis.get('suspected_metadata', []))
        to_remove.update(analysis.get('suspected_ids', []))
        
        # Add timestamp columns if configured
        preprocessing_config = config.get('preprocessing', {})
        if preprocessing_config.get('remove_timestamps', True):
            to_remove.update(analysis.get('suspected_timestamps', []))
        
        # Add address columns if configured
        if preprocessing_config.get('remove_addresses', True):
            to_remove.update(analysis.get('suspected_addresses', []))
        
        # Ensure we don't remove the target column
        to_remove.discard(target_col)
        
        # Remove only columns that exist in the dataframe
        to_remove = [col for col in to_remove if col in df.columns]
        
        if to_remove:
            df = df.drop(columns=to_remove)
            logger.info(f"Removed {len(to_remove)} redundant features: {to_remove}")
        
        return df
    
    def _convert_categorical_to_numeric(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Convert categorical features to numeric using label encoding."""
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        target_col = config.get('label_config', {}).get('target_column', 'Label')
        
        # Don't encode the target column here
        categorical_columns = [col for col in categorical_columns if col != target_col]
        
        if len(categorical_columns) > 0:
            try:
                from sklearn.preprocessing import LabelEncoder
                
                for col in categorical_columns:
                    le = LabelEncoder()
                    # Handle missing values
                    df[col] = df[col].fillna('unknown')
                    df[col] = le.fit_transform(df[col].astype(str))
                    
                logger.info(f"Encoded {len(categorical_columns)} categorical columns to numeric")
            except ImportError:
                logger.warning("sklearn not available for label encoding - keeping categorical columns as-is")
            except Exception as e:
                logger.warning(f"Categorical encoding failed: {e}")
        
        return df
    
    def _adaptive_feature_selection(self, df: pd.DataFrame, config: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Intelligent feature selection based on data analysis results."""
        
        logger.info("Starting adaptive feature selection...")
        
        target_col = config.get('label_config', {}).get('target_column', 'Label')
        feature_selection_config = config.get('feature_selection', {})
        
        # Get all feature columns (excluding target)
        all_features = [col for col in df.columns if col != target_col]
        selected_features = all_features.copy()
        
        removal_reasons = {}
        
        # 1. Remove suspected metadata and ID columns
        metadata_removal = set(analysis['suspected_metadata'] + analysis['suspected_ids'])
        if metadata_removal and feature_selection_config.get('remove_metadata', True):
            for col in metadata_removal:
                if col in selected_features:
                    selected_features.remove(col)
                    removal_reasons[col] = "Metadata/ID column"
            logger.info(f"Removed {len(metadata_removal)} metadata/ID columns")
        
        # 2. Remove timestamp columns (unless specifically requested)
        if feature_selection_config.get('remove_timestamps', True):
            for col in analysis['suspected_timestamps']:
                if col in selected_features:
                    selected_features.remove(col)
                    removal_reasons[col] = "Timestamp column"
            logger.info(f"Removed {len(analysis['suspected_timestamps'])} timestamp columns")
        
        # 3. Remove address columns (unless specifically requested)
        if feature_selection_config.get('remove_addresses', True):
            for col in analysis['suspected_addresses']:
                if col in selected_features:
                    selected_features.remove(col)
                    removal_reasons[col] = "Address column"
            logger.info(f"Removed {len(analysis['suspected_addresses'])} address columns")
        
        # 4. Remove constant/near-constant columns
        if feature_selection_config.get('remove_constant', True):
            for col in analysis['constant_columns']:
                if col in selected_features:
                    selected_features.remove(col)
                    removal_reasons[col] = "Constant/near-constant"
            logger.info(f"Removed {len(analysis['constant_columns'])} constant columns")
        
        # 5. Remove high cardinality categorical columns
        high_cardinality_threshold = feature_selection_config.get('high_cardinality_threshold', 0.9)
        if feature_selection_config.get('remove_high_cardinality', True):
            high_cardinality_removal = []
            for col in analysis['high_cardinality_columns']:
                if col in selected_features and col in analysis['categorical_columns']:
                    if analysis['unique_value_ratios'][col] > high_cardinality_threshold:
                        selected_features.remove(col)
                        removal_reasons[col] = "High cardinality categorical"
                        high_cardinality_removal.append(col)
            logger.info(f"Removed {len(high_cardinality_removal)} high cardinality categorical columns")
        
        # 6. Handle highly correlated features
        if feature_selection_config.get('remove_correlated', True) and analysis['correlation_groups']:
            corr_removal = []
            for group in analysis['correlation_groups']:
                # Keep only the first feature from each correlation group
                features_in_group = [col for col in group if col in selected_features]
                if len(features_in_group) > 1:
                    to_remove = features_in_group[1:]  # Keep first, remove rest
                    for col in to_remove:
                        selected_features.remove(col)
                        removal_reasons[col] = f"High correlation (group: {features_in_group[0]})"
                        corr_removal.append(col)
            logger.info(f"Removed {len(corr_removal)} highly correlated columns")
        
        # 7. Variance-based filtering for remaining numeric features
        if feature_selection_config.get('variance_filtering', True):
            numeric_features = [col for col in selected_features if col in analysis['numeric_columns']]
            if len(numeric_features) > 0:
                try:
                    from sklearn.feature_selection import VarianceThreshold
                    
                    variance_threshold = feature_selection_config.get('variance_threshold', 0.01)
                    numeric_df = df[numeric_features].fillna(df[numeric_features].median())
                    
                    # Normalize features for variance calculation
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    normalized_data = scaler.fit_transform(numeric_df)
                    
                    selector = VarianceThreshold(threshold=variance_threshold)
                    selector.fit(normalized_data)
                    
                    low_variance_features = [
                        numeric_features[i] for i, keep in enumerate(selector.get_support()) if not keep
                    ]
                    
                    for col in low_variance_features:
                        if col in selected_features:
                            selected_features.remove(col)
                            removal_reasons[col] = "Low variance"
                    
                    logger.info(f"Removed {len(low_variance_features)} low variance numeric columns")
                    
                except ImportError:
                    logger.warning("sklearn not available for variance filtering")
                except Exception as e:
                    logger.warning(f"Variance filtering failed: {e}")
        
        # 8. Mutual information-based selection (if target is available and sklearn is available)
        max_features = feature_selection_config.get('max_features', None)
        if max_features and len(selected_features) > max_features:
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                from sklearn.preprocessing import LabelEncoder
                
                logger.info(f"Applying mutual information selection to reduce from {len(selected_features)} to {max_features} features")
                
                # Prepare data for mutual information
                X = df[selected_features].copy()
                y = df[target_col].copy()
                
                # Handle missing values
                for col in X.select_dtypes(include=['number']).columns:
                    X[col] = X[col].fillna(X[col].median())
                
                # Encode categorical features
                label_encoders = {}
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    X[col] = X[col].fillna('unknown')
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                
                # Determine if classification or regression
                is_classification = y.dtype == 'object' or y.nunique() < 20
                
                if is_classification:
                    if y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                    mi_scores = mutual_info_classif(X, y, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                
                # Select top features based on mutual information
                feature_scores = list(zip(selected_features, mi_scores))
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top features
                top_features = [feat for feat, score in feature_scores[:max_features]]
                removed_features = [feat for feat in selected_features if feat not in top_features]
                
                for col in removed_features:
                    removal_reasons[col] = "Low mutual information"
                
                selected_features = top_features
                logger.info(f"Selected top {len(selected_features)} features based on mutual information")
                
            except ImportError:
                logger.warning("sklearn not available for mutual information selection")
            except Exception as e:
                logger.warning(f"Mutual information selection failed: {e}")
        
        # Log removal summary
        if removal_reasons:
            logger.info("Feature removal summary:")
            reason_counts = {}
            for reason in removal_reasons.values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            for reason, count in reason_counts.items():
                logger.info(f"  - {reason}: {count} features")
        
        logger.info(f"Feature selection complete: {len(all_features)} -> {len(selected_features)} features")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
    
    def _prepare_features_labels(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels from dataframe."""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        label_config = config.get('label_config', {})
        target_column = label_config.get('target_column', 'Label')
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and labels
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Convert labels to binary (normal=0, anomaly=1)
        normal_labels = label_config.get('normal_labels', ['BenignTraffic'])
        y_binary = (~y.isin(normal_labels)).astype(int)
        
        # Handle categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            le = LabelEncoder()
            # Handle unknown values
            X[col] = X[col].fillna('unknown')
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle boolean features
        boolean_columns = X.select_dtypes(include=['bool']).columns
        X[boolean_columns] = X[boolean_columns].astype(int)
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y_binary.values.astype(np.int64)
        
        # Standardize features
        preprocessing_config = config.get('preprocessing', {})
        if preprocessing_config.get('standardization', True):
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)
        
        logger.info(f"Feature matrix shape: {X_array.shape}")
        logger.info(f"Label distribution: Normal={np.sum(y_array == 0)}, Anomaly={np.sum(y_array == 1)}")
        
        return X_array, y_array
    
    def get_data_shape(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Get IoT data shape (number of features)."""
        # Try to dynamically determine features based on config
        # This is a conservative estimate - will be updated after actual loading
        preprocessing = config.get('preprocessing', {})
        exclude_columns = preprocessing.get('exclude_columns', [])
        
        # Estimate based on typical IoT datasets minus excluded columns
        estimated_features = 23 - len(exclude_columns)  # Typical total minus excluded
        return (estimated_features,)
    
    def get_num_classes(self, config: Dict[str, Any]) -> int:
        """Get number of classes for IoT data (binary classification)."""
        return 2  # Normal vs Anomaly


class IoTDataset(Dataset):
    """PyTorch dataset for IoT network traffic data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize IoT dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label vector (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        return self.features[idx], self.labels[idx]


# Register dataset loaders
DatasetRegistry.register_loader("mnist", MNISTLoader)
DatasetRegistry.register_loader("cifar10", CIFAR10Loader)
DatasetRegistry.register_loader("custom_csv", CSVLoader)
DatasetRegistry.register_loader("iot_general", IoTGeneralLoader)
