"""
Base abstract classes implementing core interfaces for TRUST-MCNet.

This module provides default implementations and common functionality
that concrete classes can inherit from, reducing code duplication
and ensuring consistent behavior across the framework.

This enhanced version includes stronger typing, persistence capabilities,
extended hooks, pluggable metrics backends, and async support.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Callable, Awaitable, AsyncGenerator
import logging
import json
import os
import sys
import time
import uuid
import asyncio
import numpy as np
import inspect
from datetime import datetime
from pathlib import Path
from functools import wraps
import random

# Optional git dependency - try to import but don't fail if not available
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False

try:
    from pydantic import BaseModel as PydanticBaseModel, validator, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    from dataclasses import dataclass
    PydanticBaseModel = object  # Fallback if pydantic not available

from .interfaces import (
    DataLoaderInterface,
    ModelInterface,
    StrategyInterface,
    TrustEvaluatorInterface,
    MetricsInterface,
    PartitionerInterface,
    ConfigInterface,
    ExperimentInterface
)
from .types import (
    ClientID,
    ModelParameters,
    Metrics,
    TrustScore,
    ClientConfig,
    ExperimentConfig,
    DatasetInfo,  # Fixed typo from DatasetI7nfo
    PartitionConfig,
    ExperimentPhase
)
from .exceptions import (
    TrustMCNetError,
    ConfigurationError,
    DataLoadingError,
    ModelError,
    TrustEvaluationError,
    PartitioningError,
    ExperimentError
)


class BaseDataLoader(DataLoaderInterface, ABC):
    """
    Base implementation for data loaders with common functionality.
    
    Features:
    - Lazy loading and streaming support
    - Built-in preprocessing and transformation pipeline
    - Automatic train/val/test splitting with stratification
    - Data validation and quality checking
    - Caching and performance optimization
    - Data visualization and exploratory analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data caching and state
        self.cache_enabled = self.config.get('cache_enabled', False)
        self.cache_dir = self.config.get('cache_dir', '.cache')
        self._data_cache = {}
        
        # Dataset properties
        self.dataset_name = self.config.get('name', 'unnamed')
        self.split_ratio = self._parse_split_ratio(self.config.get('split_ratio', '0.7,0.15,0.15'))
        self.stratify = self.config.get('stratify', False)
        self.seed = self.config.get('seed', 42)
        
        # Preprocessing config
        self.preprocessing_steps = self.config.get('preprocessing', [])
        self.augmentation_enabled = self.config.get('augmentation_enabled', False)
        self.augmentation_config = self.config.get('augmentation', {})
        
        # Streaming settings
        self.streaming_enabled = self.config.get('streaming_enabled', False)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Validate configuration before proceeding
        self._validate_config()
        
        # Setup data paths and create cache directory if needed
        self._setup_data_paths()
        
        # Initialize random number generator for reproducibility
        self.rng = random.Random(self.seed)
        try:
            import numpy as np
            self.np_rng = np.random.RandomState(self.seed)
        except ImportError:
            self.np_rng = None
    
    def _parse_split_ratio(self, split_str: Union[str, List[float]]) -> List[float]:
        """
        Parse train/val/test split ratio.
        
        Args:
            split_str: Split ratio as string "0.7,0.15,0.15" or list [0.7, 0.15, 0.15]
            
        Returns:
            List[float]: Normalized split ratios
        """
        if isinstance(split_str, str):
            try:
                ratios = [float(x) for x in split_str.split(',')]
            except ValueError:
                self.logger.warning(f"Invalid split ratio format: {split_str}, using default 0.7,0.15,0.15")
                ratios = [0.7, 0.15, 0.15]
        else:
            ratios = split_str
            
        # Validate ratios
        if sum(ratios) != 1.0:
            self.logger.warning(f"Split ratios {ratios} don't sum to 1.0, normalizing")
            total = sum(ratios)
            ratios = [r / total for r in ratios]
            
        # Ensure we have at least train and test splits
        if len(ratios) < 2:
            raise ConfigurationError("At least train and test split ratios must be provided")
            
        return ratios
        
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ConfigurationError("Config must be a dictionary")
        
        # Check required fields
        required_keys = ['name']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required config key: {key}")
                
        # Validate split ratio
        if sum(self.split_ratio) != 1.0:
            raise ConfigurationError(f"Split ratios must sum to 1.0, got {self.split_ratio}")
            
        # Validate batch size
        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")
            
        # Extended validation - subclasses should override
        self._validate_extended_config()
    
    def _validate_extended_config(self) -> None:
        """
        Extended configuration validation.
        
        Subclasses should override this method to add custom validation.
        """
        pass
    
    def _setup_data_paths(self) -> None:
        """Set up data paths and create directories if needed."""
        # Set up cache directory
        if self.cache_enabled:
            cache_dir = os.path.join(self.cache_dir, self.dataset_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.info(f"Cache directory set to {cache_dir}")
            
        # Set up data directory
        data_dir = self.config.get('data_dir')
        if data_dir:
            if not os.path.exists(data_dir):
                self.logger.warning(f"Data directory {data_dir} does not exist")
    
    def load_data(self) -> Tuple[Any, Any]:
        """
        Load training and test datasets.
        
        Returns:
            Tuple[Any, Any]: Training and test data
        """
        try:
            # Check cache first if enabled
            if self.cache_enabled:
                cached_data = self._load_from_cache()
                if cached_data:
                    self.logger.info("Loaded data from cache")
                    return cached_data
            
            # Load raw data
            raw_data = self._load_raw_data()
            
            # Preprocess data
            processed_data = self._preprocess_data(raw_data)
            
            # Split data into train/val/test
            train_data, test_data = self._split_data(processed_data)
            
            # Cache data if enabled
            if self.cache_enabled:
                self._save_to_cache((train_data, test_data))
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise DataLoadingError(f"Failed to load data: {str(e)}") from e
    
    def load_data_with_validation(self) -> Tuple[Any, Any, Any]:
        """
        Load training, validation and test datasets.
        
        Returns:
            Tuple[Any, Any, Any]: Training, validation, and test data
        """
        try:
            # Check cache first if enabled
            if self.cache_enabled:
                cached_data = self._load_from_cache('with_validation')
                if cached_data:
                    self.logger.info("Loaded data with validation split from cache")
                    return cached_data
            
            # Load raw data
            raw_data = self._load_raw_data()
            
            # Preprocess data
            processed_data = self._preprocess_data(raw_data)
            
            # Split data into train/val/test
            train_data, val_data, test_data = self._split_data_with_validation(processed_data)
            
            # Cache data if enabled
            if self.cache_enabled:
                self._save_to_cache((train_data, val_data, test_data), 'with_validation')
            
            return train_data, val_data, test_data
            
        except Exception as e:
            self.logger.error(f"Data loading with validation split failed: {str(e)}")
            raise DataLoadingError(f"Failed to load data with validation: {str(e)}") from e
    
    def load_streaming(self) -> Any:
        """
        Load data in streaming mode.
        
        Returns:
            Any: Data generator or streaming dataset
        """
        if not self.streaming_enabled:
            self.logger.warning("Streaming not enabled, falling back to standard loading")
            return self.load_data()
            
        try:
            return self._load_streaming_impl()
        except Exception as e:
            self.logger.error(f"Streaming data loading failed: {str(e)}")
            raise DataLoadingError(f"Failed to load streaming data: {str(e)}") from e
    
    def _load_streaming_impl(self) -> Any:
        """
        Implementation for streaming data loading.
        
        Returns:
            Any: Streaming dataset or generator
        """
        # Default implementation - subclasses should override
        self.logger.warning("Default streaming implementation used - not optimized")
        
        train_data, test_data = self.load_data()
        
        # Create simple generators
        def train_generator():
            indices = list(range(len(train_data)))
            self.rng.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield self._get_batch(train_data, batch_indices)
                
        def test_generator():
            indices = list(range(len(test_data)))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield self._get_batch(test_data, batch_indices)
        
        return train_generator(), test_generator()
    
    def _get_batch(self, data: Any, indices: List[int]) -> Any:
        """
        Extract a batch of data using indices.
        
        Args:
            data: Dataset
            indices: Indices to include in batch
            
        Returns:
            Any: Batch of data
        """
        # Default implementation - subclasses should override for data-specific batching
        try:
            # Common patterns
            if hasattr(data, 'iloc'):  # pandas DataFrame
                return data.iloc[indices]
            elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                # List-like object - extract batch
                if isinstance(data, tuple) and len(data) == 2:
                    # Common (X, y) pattern
                    X, y = data
                    batch_X = [X[i] for i in indices]
                    batch_y = [y[i] for i in indices]
                    return batch_X, batch_y
                else:
                    # Generic indexable object
                    return [data[i] for i in indices]
            else:
                self.logger.warning("Could not extract batch from data - unsupported type")
                return None
        except Exception as e:
            self.logger.error(f"Batch extraction failed: {str(e)}")
            return None
    
    @abstractmethod
    def _load_raw_data(self) -> Any:
        """
        Load raw data from source.
        
        Subclasses must implement this method to load data from their specific source.
        
        Returns:
            Any: Raw dataset
        """
        pass
    
    def _preprocess_data(self, data: Any) -> Any:
        """
        Apply preprocessing steps to data.
        
        Args:
            data: Raw data
            
        Returns:
            Any: Preprocessed data
        """
        if not self.preprocessing_steps:
            return data
            
        processed_data = data
        
        # Apply each preprocessing step
        for step_config in self.preprocessing_steps:
            step_name = step_config.get('name')
            step_params = step_config.get('params', {})
            
            self.logger.info(f"Applying preprocessing step: {step_name}")
            
            try:
                # Apply preprocessing step
                processed_data = self._apply_preprocessing_step(
                    step_name, processed_data, **step_params
                )
            except Exception as e:
                self.logger.error(f"Preprocessing step '{step_name}' failed: {str(e)}")
                raise DataLoadingError(f"Preprocessing step '{step_name}' failed: {str(e)}") from e
        
        return processed_data
    
    def _apply_preprocessing_step(self, step_name: str, data: Any, **params) -> Any:
        """
        Apply a specific preprocessing step.
        
        Args:
            step_name: Name of preprocessing step
            data: Input data
            **params: Step-specific parameters
            
        Returns:
            Any: Processed data
        """
        # Check for built-in preprocessing steps
        if hasattr(self, f"_preprocess_{step_name}"):
            method = getattr(self, f"_preprocess_{step_name}")
            return method(data, **params)
        else:
            # Look for custom preprocessing in subclasses
            self.logger.warning(f"Unknown preprocessing step: {step_name}")
            return data
    
    def _preprocess_normalize(self, data: Any, method: str = 'minmax', **params) -> Any:
        """
        Normalize data values.
        
        Args:
            data: Input data
            method: Normalization method ('minmax', 'zscore', etc.)
            **params: Method-specific parameters
            
        Returns:
            Any: Normalized data
        """
        try:
            import numpy as np
            
            # Handle common data formats
            if isinstance(data, tuple) and len(data) == 2:
                # Handle (features, labels) format
                X, y = data
                if isinstance(X, np.ndarray):
                    if method == 'minmax':
                        X_min = X.min(axis=0)
                        X_max = X.max(axis=0)
                        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
                    elif method == 'zscore':
                        X_mean = X.mean(axis=0)
                        X_std = X.std(axis=0) + 1e-8
                        X_norm = (X - X_mean) / X_std
                    else:
                        self.logger.warning(f"Unknown normalization method: {method}")
                        X_norm = X
                    return X_norm, y
                else:
                    self.logger.warning("Cannot normalize non-numpy array data")
                    return data
            elif isinstance(data, np.ndarray):
                # Handle numpy array directly
                if method == 'minmax':
                    data_min = data.min(axis=0)
                    data_max = data.max(axis=0)
                    return (data - data_min) / (data_max - data_min + 1e-8)
                elif method == 'zscore':
                    data_mean = data.mean(axis=0)
                    data_std = data.std(axis=0) + 1e-8
                    return (data - data_mean) / data_std
                else:
                    self.logger.warning(f"Unknown normalization method: {method}")
                    return data
            else:
                self.logger.warning(f"Cannot normalize data of type: {type(data)}")
                return data
                
        except ImportError:
            self.logger.warning("NumPy required for normalization")
            return data
        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}")
            return data
    
    def _preprocess_fillna(self, data: Any, strategy: str = 'mean', **params) -> Any:
        """
        Fill missing values in data.
        
        Args:
            data: Input data
            strategy: Fill strategy ('mean', 'median', 'zero', 'value')
            **params: Strategy-specific parameters
            
        Returns:
            Any: Data with missing values filled
        """
        try:
            import numpy as np
            
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                if strategy == 'mean':
                    col_means = np.nanmean(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(col_means, inds[1])
                elif strategy == 'median':
                    col_medians = np.nanmedian(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(col_medians, inds[1])
                elif strategy == 'zero':
                    data = np.nan_to_num(data)
                elif strategy == 'value':
                    fill_value = params.get('value', 0)
                    data = np.nan_to_num(data, nan=fill_value)
                return data
                
            # Handle pandas DataFrames
            elif hasattr(data, 'fillna'):
                if strategy == 'mean':
                    return data.fillna(data.mean())
                elif strategy == 'median':
                    return data.fillna(data.median())
                elif strategy == 'zero':
                    return data.fillna(0)
                elif strategy == 'value':
                    fill_value = params.get('value', 0)
                    return data.fillna(fill_value)
                return data
                
            # Handle (X, y) format
            elif isinstance(data, tuple) and len(data) == 2:
                X, y = data
                return self._preprocess_fillna(X, strategy, **params), y
                
            else:
                self.logger.warning(f"Cannot fill NAs in data of type: {type(data)}")
                return data
                
        except ImportError:
            self.logger.warning("Required libraries missing for fillna")
            return data
        except Exception as e:
            self.logger.error(f"Fill NA failed: {str(e)}")
            return data
    
    def _split_data(self, data: Any) -> Tuple[Any, Any]:
        """
        Split data into training and test sets.
        
        Args:
            data: Input data
            
        Returns:
            Tuple[Any, Any]: Training and test data
        """
        # Calculate split indices
        train_ratio = self.split_ratio[0]
        
        # If we have 3 values, combine validation into training for this method
        if len(self.split_ratio) >= 3:
            train_ratio += self.split_ratio[1]
            
        return self._split_data_impl(data, [train_ratio, 1-train_ratio])
    
    def _split_data_with_validation(self, data: Any) -> Tuple[Any, Any, Any]:
        """
        Split data into training, validation and test sets.
        
        Args:
            data: Input data
            
        Returns:
            Tuple[Any, Any, Any]: Training, validation and test data
        """
        # Ensure we have at least 3 split values
        split_ratios = self.split_ratio
        if len(split_ratios) < 3:
            # If only 2 provided, use default validation split
            train_ratio, test_ratio = split_ratios
            val_ratio = test_ratio * 0.5
            test_ratio = test_ratio * 0.5
            split_ratios = [train_ratio, val_ratio, test_ratio]
            
            # Normalize to ensure they sum to 1
            total = sum(split_ratios)
            split_ratios = [r / total for r in split_ratios]
            
        # Apply the split
        return self._split_data_impl(data, split_ratios)
        
    def _split_data_impl(self, data: Any, split_ratios: List[float]) -> Tuple[Any, ...]:
        """
        Implementation for data splitting.
        
        Args:
            data: Input data
            split_ratios: List of split ratios (must sum to 1)
            
        Returns:
            Tuple[Any, ...]: Split datasets
        """
        try:
            # Different handling based on data type
            if hasattr(data, 'shape') and hasattr(data, 'iloc'):
                # Pandas DataFrame/Series
                return self._split_dataframe(data, split_ratios)
            elif hasattr(data, 'shape') and hasattr(data, '__array__'):
                # NumPy array
                return self._split_numpy(data, split_ratios)
            elif isinstance(data, tuple) and len(data) == 2:
                # Common (X, y) format
                return self._split_xy_tuple(data, split_ratios)
            else:
                # Try generic list-like splitting
                return self._split_generic(data, split_ratios)
                
        except Exception as e:
            self.logger.error(f"Data splitting failed: {str(e)}")
            raise DataLoadingError(f"Failed to split data: {str(e)}") from e
    
    def _split_dataframe(self, df: Any, split_ratios: List[float]) -> Tuple[Any, ...]:
        """
        Split pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            split_ratios: List of split ratios
            
        Returns:
            Tuple[Any, ...]: Split DataFrames
        """
        n = len(df)
        indices = list(range(n))
        
        # Shuffle indices
        self.rng.shuffle(indices)
        
        # Calculate split sizes
        split_sizes = [int(r * n) for r in split_ratios[:-1]]
        split_sizes.append(n - sum(split_sizes))  # Ensure all data is used
        
        # Split indices
        result = []
        start = 0
        for size in split_sizes:
            split_indices = indices[start:start+size]
            result.append(df.iloc[split_indices])
            start += size
            
        return tuple(result)
    
    def _split_numpy(self, arr: Any, split_ratios: List[float]) -> Tuple[Any, ...]:
        """
        Split numpy array.
        
        Args:
            arr: NumPy array
            split_ratios: List of split ratios
            
        Returns:
            Tuple[Any, ...]: Split arrays
        """
        import numpy as np
        
        n = len(arr)
        indices = np.arange(n)
        
        # Shuffle indices
        if self.np_rng is not None:
            self.np_rng.shuffle(indices)
        else:
            # Use Python's random if NumPy RNG not available
            indices = list(indices)
            self.rng.shuffle(indices)
            indices = np.array(indices)
        
        # Calculate split sizes
        split_sizes = [int(r * n) for r in split_ratios[:-1]]
        split_sizes.append(n - sum(split_sizes))  # Ensure all data is used
        
        # Split indices
        result = []
        start = 0
        for size in split_sizes:
            split_indices = indices[start:start+size]
            result.append(arr[split_indices])
            start += size
            
        return tuple(result)
    
    def _split_xy_tuple(self, data: Tuple[Any, Any], split_ratios: List[float]) -> Tuple[Tuple[Any, Any], ...]:
        """
        Split (X, y) data tuple.
        
        Args:
            data: (X, y) tuple
            split_ratios: List of split ratios
            
        Returns:
            Tuple[Tuple[Any, Any], ...]: Split (X, y) tuples
        """
        X, y = data
        
        # Get appropriate splitting function
        if hasattr(X, 'shape') and hasattr(X, 'iloc'):
            # Pandas DataFrame/Series
            split_func = self._split_dataframe
        elif hasattr(X, 'shape') and hasattr(X, '__array__'):
            # NumPy array
            split_func = self._split_numpy
        else:
            # Generic list-like
            split_func = self._split_generic
            
        # Generate common indices for both X and y
        n = len(X)
        indices = list(range(n))
        
        # Shuffle indices
        self.rng.shuffle(indices)
        
        # Calculate split sizes
        split_sizes = [int(r * n) for r in split_ratios[:-1]]
        split_sizes.append(n - sum(split_sizes))  # Ensure all data is used
        
        # Apply splits with same indices to both X and y
        result = []
        start = 0
        for size in split_sizes:
            split_indices = indices[start:start+size]
            
            # Extract X and y subsets
            if hasattr(X, 'iloc'):
                X_split = X.iloc[split_indices]
            elif hasattr(X, '__array__'):
                X_split = X[split_indices]
            else:
                X_split = [X[i] for i in split_indices]
                
            if hasattr(y, 'iloc'):
                y_split = y.iloc[split_indices]
            elif hasattr(y, '__array__'):
                y_split = y[split_indices]
            else:
                y_split = [y[i] for i in split_indices]
                
            result.append((X_split, y_split))
            start += size
            
        return tuple(result)
    
    def _split_generic(self, data: Any, split_ratios: List[float]) -> Tuple[Any, ...]:
        """
        Split generic list-like data.
        
        Args:
            data: List-like data
            split_ratios: List of split ratios
            
        Returns:
            Tuple[Any, ...]: Split datasets
        """
        n = len(data)
        indices = list(range(n))
        
        # Shuffle indices
        self.rng.shuffle(indices)
        
        # Calculate split sizes
        split_sizes = [int(r * n) for r in split_ratios[:-1]]
        split_sizes.append(n - sum(split_sizes))  # Ensure all data is used
        
        # Split indices
        result = []
        start = 0
        for size in split_sizes:
            split_indices = indices[start:start+size]
            result.append([data[i] for i in split_indices])
            start += size
            
        return tuple(result)
    
    def _load_from_cache(self, cache_key: str = 'default') -> Optional[Tuple[Any, ...]]:
        """
        Load data from cache.
        
        Args:
            cache_key: Cache identifier
            
        Returns:
            Optional[Tuple[Any, ...]]: Cached data or None if not found/valid
        """
        if not self.cache_enabled:
            return None
            
        # Generate cache path
        cache_file = os.path.join(
            self.cache_dir,
            self.dataset_name,
            f"{cache_key}_{self._get_cache_key_suffix()}.pkl"
        )
        
        # Check if cache file exists and is valid
        if not os.path.exists(cache_file):
            return None
            
        try:
            # Load from cache
            with open(cache_file, 'rb') as f:
                import pickle
                cached_data = pickle.load(f)
                
            # Validate cache
            if self._validate_cache(cached_data):
                self.logger.info(f"Loaded data from cache: {cache_file}")
                return cached_data
            else:
                self.logger.warning(f"Invalid cache: {cache_file}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, data: Tuple[Any, ...], cache_key: str = 'default') -> None:
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            cache_key: Cache identifier
        """
        if not self.cache_enabled:
            return
            
        # Generate cache path
        cache_dir = os.path.join(self.cache_dir, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(
            cache_dir, 
            f"{cache_key}_{self._get_cache_key_suffix()}.pkl"
        )
        
        try:
            # Save to cache
            with open(cache_file, 'wb') as f:
                import pickle
                pickle.dump(data, f)
                
            self.logger.info(f"Saved data to cache: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {str(e)}")
    
    def _get_cache_key_suffix(self) -> str:
        """
        Generate cache key suffix based on configuration.
        
        Returns:
            str: Cache key suffix
        """
        # Create deterministic key from config
        key_components = [
            f"split_{'-'.join(str(r) for r in self.split_ratio)}",
            f"seed_{self.seed}"
        ]
        
        # Add preprocessing steps to key
        if self.preprocessing_steps:
            prep_key = "_".join(step['name'] for step in self.preprocessing_steps)
            key_components.append(f"prep_{prep_key}")
            
        return "_".join(key_components)
    
    def _validate_cache(self, cached_data: Any) -> bool:
        """
        Validate cached data.
        
        Args:
            cached_data: Data loaded from cache
            
        Returns:
            bool: True if cache is valid
        """
        # Basic validation
        if not isinstance(cached_data, tuple):
            return False
            
        # Check if the number of splits matches expected
        if len(self.split_ratio) == 2 and len(cached_data) != 2:
            return False
        elif len(self.split_ratio) >= 3 and len(cached_data) != 3:
            return False
            
        # Additional validation - subclasses can override
        return True
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information and statistics.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        info = {
            'name': self.dataset_name,
            'config': {
                'split_ratio': self.split_ratio,
                'seed': self.seed,
                'cache_enabled': self.cache_enabled
            }
        }
        
        # Add dataset-specific information if available
        dataset_info = self._get_dataset_specific_info()
        if dataset_info:
            info.update(dataset_info)
            
        return info
    
    def _get_dataset_specific_info(self) -> Dict[str, Any]:
        """
        Get dataset-specific information and statistics.
        
        Subclasses should override to provide dataset-specific information.
        
        Returns:
            Dict[str, Any]: Dataset-specific information
        """
        return {}
    
    @abstractmethod
    def get_data_info(self) -> DatasetInfo:
        """Get information about the dataset."""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """Validate loaded data."""
        if data is None:
            return False
        
        # Basic validation - subclasses can override for specific checks
        try:
            len(data)
            return True
        except (TypeError, AttributeError):
            self.logger.warning("Data validation failed: unable to get length")
            return False


class BaseModel(ModelInterface, ABC):
    """
    Base implementation for models with common functionality.
    
    Provides parameter handling, validation, checkpoint management,
    template training loops, and utility methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = self.config.get('epochs', 1)
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_frequency = self.config.get('checkpoint_frequency', 5)
        self.model_id = f"{self.config.get('type', 'model')}_{uuid.uuid4().hex[:8]}"
        self._validate_config()
        
        # Create checkpoint directory if it doesn't exist
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Model config must be a dictionary")
        
        required_keys = ['type']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required model config key: {key}")
    
    @abstractmethod
    def get_parameters(self) -> ModelParameters:
        """
        Get model parameters.
        
        Returns:
            ModelParameters: The current model parameters.
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: ModelParameters) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: New model parameters to set.
        """
        pass
    
    def train(self, data: Any) -> Metrics:
        """
        Train the model using a template pattern.
        
        This implementation provides a standardized training loop that:
        1. Tracks metrics across epochs
        2. Handles checkpointing
        3. Calls customizable hooks for specific model implementations
        
        Args:
            data: Training data in format required by model implementation
            
        Returns:
            Metrics: Training metrics including loss, accuracy, etc.
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()
        all_metrics = {}
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Call implementation-specific training logic
            epoch_metrics = self._train_epoch(data, epoch)
            
            epoch_time = time.time() - epoch_start
            epoch_metrics['time'] = epoch_time
            
            # Store metrics
            all_metrics[f"epoch_{epoch+1}"] = epoch_metrics
            
            # Call post-epoch hook for custom processing
            self._post_epoch_hook(epoch, epoch_metrics)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_time:.2f}s")
            
            # Create checkpoint if needed
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(f"{self.model_id}_epoch_{epoch+1}")
        
        # Final metrics
        training_time = time.time() - start_time
        all_metrics['total_time'] = training_time
        all_metrics['average_epoch_time'] = training_time / self.epochs
        
        self.logger.info(f"Training completed in {training_time:.2f}s")
        return all_metrics
    
    @abstractmethod
    def _train_epoch(self, data: Any, epoch: int) -> Dict[str, Any]:
        """
        Implementation-specific logic for training a single epoch.
        
        Args:
            data: Training data
            epoch: Current epoch number
            
        Returns:
            Dict[str, Any]: Metrics for this epoch
        """
        pass
    
    def _post_epoch_hook(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Hook called after each epoch for customizable processing.
        
        Subclasses can override this to implement learning rate scheduling,
        early stopping, or other epoch-dependent logic.
        
        Args:
            epoch: Current epoch number
            metrics: Metrics from the current epoch
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: Any) -> Metrics:
        """
        Evaluate the model.
        
        Args:
            data: Evaluation data
            
        Returns:
            Metrics: Evaluation metrics
        """
        pass
    
    def save_checkpoint(self, checkpoint_name: str = None) -> Path:
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_name: Optional name for the checkpoint.
                If not provided, a timestamped name is generated.
                
        Returns:
            Path: Path to the saved checkpoint file
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{self.model_id}_{timestamp}"
            
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.checkpoint"
        
        try:
            checkpoint_data = {
                'parameters': self.get_parameters(),
                'config': self.config,
                'model_id': self.model_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': self._get_checkpoint_metadata()
            }
            
            # Implementation-specific state saving
            model_state = self._get_model_state()
            if model_state:
                checkpoint_data['model_state'] = model_state
                
            # Save the checkpoint
            self._save_checkpoint_file(checkpoint_path, checkpoint_data)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            # Load the checkpoint
            checkpoint_data = self._load_checkpoint_file(checkpoint_path)
            
            # Restore parameters
            if 'parameters' in checkpoint_data:
                self.set_parameters(checkpoint_data['parameters'])
                
            # Restore model-specific state
            if 'model_state' in checkpoint_data:
                self._restore_model_state(checkpoint_data['model_state'])
                
            # Update model info
            if 'model_id' in checkpoint_data:
                self.model_id = checkpoint_data['model_id']
                
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for checkpointing.
        
        Subclasses should override to include optimizer state,
        learning rate scheduler state, etc.
        
        Returns:
            Dict[str, Any]: Model-specific state
        """
        return {}
    
    def _restore_model_state(self, state: Dict[str, Any]) -> None:
        """
        Restore model-specific state from checkpoint.
        
        Subclasses should override to restore optimizer state,
        learning rate scheduler state, etc.
        
        Args:
            state: Model-specific state from checkpoint
        """
        pass
    
    def _get_checkpoint_metadata(self) -> Dict[str, Any]:
        """
        Get metadata to include in checkpoint.
        
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        return {
            'framework_version': '1.0.0',  # Replace with actual version
            'python_version': sys.version,
        }
    
    def _save_checkpoint_file(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Save checkpoint data to file.
        
        Default implementation uses JSON. Subclasses may override
        to use pickle, torch.save, etc.
        
        Args:
            path: Path to save checkpoint
            data: Checkpoint data
        """
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def _load_checkpoint_file(self, path: Path) -> Dict[str, Any]:
        """
        Load checkpoint data from file.
        
        Default implementation uses JSON. Subclasses may override
        to use pickle, torch.load, etc.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'type': self.config.get('type'),
            'config': self.config,
            'model_id': self.model_id,
            'parameters_count': self._count_parameters(),
            'checkpoint_dir': str(self.checkpoint_dir),
            'has_checkpoint': self._has_checkpoint(),
        }
        
    def _count_parameters(self) -> int:
        """
        Count model parameters.
        
        Subclasses should override for accurate parameter counting.
        
        Returns:
            int: Number of trainable parameters
        """
        return 0
        
    def _has_checkpoint(self) -> bool:
        """
        Check if model has checkpoints.
        
        Returns:
            bool: True if checkpoints exist for this model
        """
        try:
            checkpoints = list(self.checkpoint_dir.glob(f"{self.model_id}*.checkpoint"))
            return len(checkpoints) > 0
        except Exception:
            return False


class BaseStrategy(StrategyInterface, ABC):
    """
    Base implementation for federated learning strategies.
    
    Provides common aggregation logic and client management with hooks for
    customizing the aggregation pipeline and client selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy configuration parameters
        self.client_fraction = self.config.get('client_fraction', 1.0)
        self.min_clients = self.config.get('min_clients', 1)
        self.max_clients = self.config.get('max_clients', float('inf'))
        self.client_selection_seed = self.config.get('client_selection_seed', None)
        self.aggregation_timeout = self.config.get('aggregation_timeout', 60)  # seconds
        
        # Pre-registered hooks
        self.pre_aggregate_hooks = []
        self.post_aggregate_hooks = []
        self.pre_client_selection_hooks = []
        self.post_client_selection_hooks = []
        
        # Strategy state
        self.current_round = 0
        self.aggregation_history = []
        self.selection_history = []
        
        self._validate_config()
        
        # Set random seed if specified
        if self.client_selection_seed is not None:
            random.seed(self.client_selection_seed)
    
    def _validate_config(self) -> None:
        """Validate strategy configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Strategy config must be a dictionary")
        
        required_keys = ['name']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required strategy config key: {key}")
                
        # Validate fraction
        if not 0.0 < self.client_fraction <= 1.0:
            raise ConfigurationError(f"client_fraction must be between 0 and 1, got {self.client_fraction}")
            
        # Validate min clients
        if self.min_clients < 1:
            raise ConfigurationError(f"min_clients must be at least 1, got {self.min_clients}")
    
    def aggregate(self, client_updates: List[Tuple[ClientID, ModelParameters, Metrics]]) -> ModelParameters:
        """
        Aggregate client updates using a pipeline of pre/post hooks.
        
        This template method defines the aggregation workflow:
        1. Pre-aggregation hooks for filtering and validating updates
        2. Core aggregation algorithm (implemented by subclasses)
        3. Post-aggregation hooks for finalizing global model
        
        Args:
            client_updates: List of (client_id, parameters, metrics) tuples
            
        Returns:
            ModelParameters: Aggregated global model parameters
        """
        self.current_round += 1
        self.logger.info(f"Round {self.current_round}: Aggregating updates from {len(client_updates)} clients")
        
        # Record aggregation metadata
        aggregation_meta = {
            'round': self.current_round,
            'client_count': len(client_updates),
            'client_ids': [client_id for client_id, _, _ in client_updates],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Apply pre-aggregation hooks
        start_time = time.time()
        try:
            filtered_updates = self._apply_pre_aggregate_hooks(client_updates)
            
            # Core aggregation (implemented by subclasses)
            if not filtered_updates:
                self.logger.warning("No valid updates after pre-aggregation filtering")
                # If all updates were filtered out, return None or last known good model
                aggregated_parameters = self._handle_empty_aggregation()
            else:
                aggregated_parameters = self._aggregate_impl(filtered_updates)
            
            # Apply post-aggregation hooks
            final_parameters = self._apply_post_aggregate_hooks(aggregated_parameters, client_updates)
            
            # Record success
            aggregation_time = time.time() - start_time
            aggregation_meta.update({
                'status': 'success',
                'duration': aggregation_time,
                'filtered_count': len(client_updates) - len(filtered_updates),
            })
            
            self.logger.info(f"Aggregation completed in {aggregation_time:.2f}s")
            return final_parameters
            
        except Exception as e:
            # Record failure
            aggregation_time = time.time() - start_time
            aggregation_meta.update({
                'status': 'failure',
                'duration': aggregation_time,
                'error': str(e),
            })
            
            self.logger.error(f"Aggregation failed: {str(e)}")
            raise
        
        finally:
            # Always record aggregation history
            self.aggregation_history.append(aggregation_meta)
    
    @abstractmethod
    def _aggregate_impl(self, client_updates: List[Tuple[ClientID, ModelParameters, Metrics]]) -> ModelParameters:
        """
        Implementation-specific aggregation logic.
        
        Args:
            client_updates: Pre-filtered client updates
            
        Returns:
            ModelParameters: Aggregated parameters
        """
        pass
    
    def _handle_empty_aggregation(self) -> ModelParameters:
        """
        Handle the case when no valid updates are available for aggregation.
        
        Subclasses can override this to implement fallback strategies.
        
        Returns:
            ModelParameters: Fallback model parameters
        """
        raise RuntimeError("No valid client updates available for aggregation")
    
    def select_clients(self, available_clients: List[ClientID]) -> List[ClientID]:
        """
        Select clients for the next round.
        
        Default implementation uses random sampling based on client_fraction.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            List[ClientID]: Selected client IDs
        """
        # Apply pre-selection hooks
        filtered_clients = self._apply_pre_selection_hooks(available_clients)
        
        # Determine number of clients to select
        num_clients = max(
            self.min_clients,
            min(
                self.max_clients,
                int(len(filtered_clients) * self.client_fraction)
            )
        )
        
        # Ensure we don't select more clients than available
        num_clients = min(num_clients, len(filtered_clients))
        
        # Select clients
        selected_clients = self._select_clients_impl(filtered_clients, num_clients)
        
        # Apply post-selection hooks
        final_selected = self._apply_post_selection_hooks(selected_clients, available_clients)
        
        # Record selection
        selection_meta = {
            'round': self.current_round + 1,  # Next round
            'available_count': len(available_clients),
            'filtered_count': len(available_clients) - len(filtered_clients),
            'selected_count': len(final_selected),
            'selected_ids': final_selected,
            'timestamp': datetime.now().isoformat(),
        }
        self.selection_history.append(selection_meta)
        
        self.logger.info(f"Selected {len(final_selected)} clients from {len(available_clients)} available")
        return final_selected
    
    def _select_clients_impl(self, clients: List[ClientID], num_clients: int) -> List[ClientID]:
        """
        Implementation-specific client selection logic.
        
        Default implementation uses random selection.
        Subclasses can override for more sophisticated selection.
        
        Args:
            clients: Available client IDs after filtering
            num_clients: Number of clients to select
            
        Returns:
            List[ClientID]: Selected client IDs
        """
        if num_clients >= len(clients):
            return clients.copy()
        return random.sample(clients, num_clients)
    
    def _apply_pre_aggregate_hooks(self, client_updates: List[Tuple[ClientID, ModelParameters, Metrics]]) -> List[Tuple[ClientID, ModelParameters, Metrics]]:
        """
        Apply pre-aggregation hooks to filter and process client updates.
        
        Args:
            client_updates: Raw client updates
            
        Returns:
            List[Tuple[ClientID, ModelParameters, Metrics]]: Filtered updates
        """
        filtered_updates = client_updates
        for hook in self.pre_aggregate_hooks:
            filtered_updates = hook(filtered_updates, {'round': self.current_round})
            
            # Check if hook removed all updates
            if not filtered_updates:
                self.logger.warning(f"Pre-aggregation hook {hook.__name__} filtered out all updates")
                break
                
        return filtered_updates
    
    def _apply_post_aggregate_hooks(self, aggregated_parameters: ModelParameters, 
                                  original_updates: List[Tuple[ClientID, ModelParameters, Metrics]]) -> ModelParameters:
        """
        Apply post-aggregation hooks to process the aggregated parameters.
        
        Args:
            aggregated_parameters: Aggregated model parameters
            original_updates: Original client updates for context
            
        Returns:
            ModelParameters: Processed global parameters
        """
        processed_parameters = aggregated_parameters
        for hook in self.post_aggregate_hooks:
            processed_parameters = hook(processed_parameters, {
                'round': self.current_round,
                'client_updates': original_updates
            })
            
        return processed_parameters
    
    def _apply_pre_selection_hooks(self, available_clients: List[ClientID]) -> List[ClientID]:
        """
        Apply pre-selection hooks to filter available clients.
        
        Args:
            available_clients: All available clients
            
        Returns:
            List[ClientID]: Filtered client list
        """
        filtered_clients = available_clients
        for hook in self.pre_client_selection_hooks:
            filtered_clients = hook(filtered_clients, {'round': self.current_round + 1})
            
        return filtered_clients
    
    def _apply_post_selection_hooks(self, selected_clients: List[ClientID], 
                                  original_clients: List[ClientID]) -> List[ClientID]:
        """
        Apply post-selection hooks to finalize client selection.
        
        Args:
            selected_clients: Initially selected clients
            original_clients: All available clients for context
            
        Returns:
            List[ClientID]: Final selected clients
        """
        final_clients = selected_clients
        for hook in self.post_client_selection_hooks:
            final_clients = hook(final_clients, {
                'round': self.current_round + 1,
                'available_clients': original_clients
            })
            
        return final_clients
    
    def register_pre_aggregate_hook(self, hook: Callable[[List[Tuple[ClientID, ModelParameters, Metrics]], Dict[str, Any]], 
                                                      List[Tuple[ClientID, ModelParameters, Metrics]]]) -> None:
        """
        Register a hook to run before aggregation.
        
        Hooks should filter or modify client updates before aggregation.
        
        Args:
            hook: Function taking (updates, context) and returning filtered updates
        """
        self.pre_aggregate_hooks.append(hook)
        
    def register_post_aggregate_hook(self, hook: Callable[[ModelParameters, Dict[str, Any]], ModelParameters]) -> None:
        """
        Register a hook to run after aggregation.
        
        Hooks should process the aggregated model parameters.
        
        Args:
            hook: Function taking (parameters, context) and returning processed parameters
        """
        self.post_aggregate_hooks.append(hook)
        
    def register_pre_selection_hook(self, hook: Callable[[List[ClientID], Dict[str, Any]], List[ClientID]]) -> None:
        """
        Register a hook to run before client selection.
        
        Hooks should filter available clients before selection.
        
        Args:
            hook: Function taking (clients, context) and returning filtered clients
        """
        self.pre_client_selection_hooks.append(hook)
        
    def register_post_selection_hook(self, hook: Callable[[List[ClientID], Dict[str, Any]], List[ClientID]]) -> None:
        """
        Register a hook to run after client selection.
        
        Hooks should finalize client selection.
        
        Args:
            hook: Function taking (selected, context) and returning final selection
        """
        self.post_client_selection_hooks.append(hook)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the strategy.
        
        Returns:
            Dict[str, Any]: Strategy information and state
        """
        return {
            'name': self.config.get('name'),
            'config': self.config,
            'current_round': self.current_round,
            'aggregation_history_size': len(self.aggregation_history),
            'selection_history_size': len(self.selection_history),
            'pre_aggregate_hooks': [hook.__name__ for hook in self.pre_aggregate_hooks],
            'post_aggregate_hooks': [hook.__name__ for hook in self.post_aggregate_hooks],
            'pre_selection_hooks': [hook.__name__ for hook in self.pre_client_selection_hooks],
            'post_selection_hooks': [hook.__name__ for hook in self.post_client_selection_hooks],
        }


TrustEvaluatorFn = Callable[[ClientID, ModelParameters, Dict[str, Any]], float]

class BaseTrustEvaluator(TrustEvaluatorInterface, ABC):
    """
    Base implementation for trust evaluators.
    
    Features:
    - Dynamic thresholding based on historical trust scores
    - Composable evaluator functions
    - Comprehensive client history tracking
    - Anomaly detection and pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Client history tracking
        self.client_history: Dict[ClientID, List[Dict[str, Any]]] = {}
        self.global_trust_history: List[Dict[str, Any]] = []
        
        # Trust threshold configuration
        self.static_threshold = self.config.get('threshold')
        self.use_dynamic_threshold = self.config.get('use_dynamic_threshold', False)
        self.percentile_threshold = self.config.get('percentile_threshold', 25)  # Bottom 25% are untrusted
        self.window_size = self.config.get('window_size', 5)  # Rounds for dynamic threshold
        
        # Evaluator function composition
        self.evaluator_weights = self.config.get('evaluator_weights', {})
        self.evaluator_functions: Dict[str, TrustEvaluatorFn] = {}
        
        # Anomaly detection
        self.anomaly_detection_enabled = self.config.get('anomaly_detection', False)
        self.anomaly_sensitivity = self.config.get('anomaly_sensitivity', 2.0)  # Standard deviations
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate trust evaluator configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Trust evaluator config must be a dictionary")
        
        # Validate static threshold if present
        if self.static_threshold is not None:
            if not (0.0 <= self.static_threshold <= 1.0):
                raise ConfigurationError(
                    f"Trust threshold must be between 0.0 and 1.0, got {self.static_threshold}"
                )
        
        # Validate percentile threshold
        if not (0 <= self.percentile_threshold <= 100):
            raise ConfigurationError(
                f"Percentile threshold must be between 0 and 100, got {self.percentile_threshold}"
            )
            
        # Validate window size
        if self.window_size < 1:
            raise ConfigurationError(
                f"Window size must be at least 1, got {self.window_size}"
            )
            
        # Validate evaluator weights if present
        if self.evaluator_weights:
            total_weight = sum(self.evaluator_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                raise ConfigurationError(
                    f"Evaluator weights must sum to 1.0, got {total_weight}"
                )
    
    def evaluate_trust(self, client_id: ClientID, model_update: ModelParameters, 
                      context: Dict[str, Any]) -> TrustScore:
        """
        Evaluate trust score for a client using a pipeline of evaluators.
        
        Args:
            client_id: Client identifier
            model_update: Client model update parameters
            context: Additional context for evaluation
            
        Returns:
            TrustScore: Composite trust score with details
        """
        start_time = time.time()
        round_num = context.get('round', 0)
        
        # Prepare result structure
        trust_result = {
            'client_id': client_id,
            'timestamp': datetime.now().isoformat(),
            'round': round_num,
            'individual_scores': {},
            'flags': [],
        }
        
        # Use registered evaluator functions if available
        if self.evaluator_functions:
            trust_result['individual_scores'] = self._run_evaluator_functions(
                client_id, model_update, context
            )
            trust_result['score'] = self._combine_evaluator_scores(trust_result['individual_scores'])
        else:
            # Use implementation-specific logic
            trust_result['score'] = self._evaluate_trust_impl(client_id, model_update, context)
        
        # Check for anomalies in client behavior
        if self.anomaly_detection_enabled:
            anomalies = self._detect_anomalies(client_id, trust_result['score'], context)
            if anomalies:
                trust_result['flags'].extend(anomalies)
        
        # Determine trusted status using appropriate threshold
        threshold = self._get_trust_threshold(round_num)
        trust_result['threshold'] = threshold
        trust_result['trusted'] = trust_result['score'] >= threshold
        
        # Add evaluation time
        trust_result['evaluation_time'] = time.time() - start_time
        
        # Update history
        self.update_client_history(client_id, trust_result)
        self.global_trust_history.append({
            'client_id': client_id,
            'round': round_num,
            'score': trust_result['score'],
            'threshold': threshold,
            'trusted': trust_result['trusted'],
        })
        
        self.logger.info(
            f"Client {client_id}: Trust score {trust_result['score']:.4f} "
            f"(threshold: {threshold:.4f}), trusted: {trust_result['trusted']}"
        )
        
        return trust_result
    
    @abstractmethod
    def _evaluate_trust_impl(self, client_id: ClientID, model_update: ModelParameters, 
                           context: Dict[str, Any]) -> float:
        """
        Implementation-specific trust evaluation logic.
        
        Args:
            client_id: Client identifier
            model_update: Client model parameters
            context: Additional context for evaluation
            
        Returns:
            float: Trust score between 0.0 and 1.0
        """
        pass
    
    def register_evaluator(self, name: str, evaluator_fn: TrustEvaluatorFn, weight: float = None) -> None:
        """
        Register a trust evaluator function.
        
        Args:
            name: Unique identifier for this evaluator
            evaluator_fn: Function that computes a trust score
            weight: Optional weight for this evaluator in composite score
        """
        if name in self.evaluator_functions:
            raise ValueError(f"Evaluator '{name}' already registered")
            
        self.evaluator_functions[name] = evaluator_fn
        
        # Update weights if provided
        if weight is not None:
            self.evaluator_weights[name] = weight
            
            # Normalize weights
            total = sum(self.evaluator_weights.values())
            for key in self.evaluator_weights:
                self.evaluator_weights[key] /= total
                
        self.logger.info(f"Registered trust evaluator: {name}")
    
    def _run_evaluator_functions(self, client_id: ClientID, model_update: ModelParameters,
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Run all registered evaluator functions.
        
        Args:
            client_id: Client identifier
            model_update: Client model parameters
            context: Additional context for evaluation
            
        Returns:
            Dict[str, float]: Scores for each evaluator
        """
        scores = {}
        
        for name, func in self.evaluator_functions.items():
            try:
                score = func(client_id, model_update, context)
                scores[name] = max(0.0, min(1.0, score))  # Clamp to [0,1]
            except Exception as e:
                self.logger.error(f"Error in evaluator '{name}': {str(e)}")
                scores[name] = 0.0  # Default to untrusted on error
                
        return scores
    
    def _combine_evaluator_scores(self, scores: Dict[str, float]) -> float:
        """
        Combine individual evaluator scores into a composite score.
        
        Args:
            scores: Individual scores from each evaluator
            
        Returns:
            float: Composite trust score
        """
        if not scores:
            return 0.0
            
        # Use weights if available, otherwise use average
        if self.evaluator_weights:
            total_score = 0.0
            total_weight = 0.0
            
            for name, score in scores.items():
                weight = self.evaluator_weights.get(name, 0.0)
                total_score += score * weight
                total_weight += weight
                
            if total_weight > 0:
                return total_score / total_weight
            else:
                return sum(scores.values()) / len(scores)
        else:
            return sum(scores.values()) / len(scores)
    
    def _get_trust_threshold(self, round_num: int) -> float:
        """
        Get the appropriate trust threshold for the current round.
        
        Uses dynamic thresholding if enabled, otherwise static threshold.
        
        Args:
            round_num: Current federation round number
            
        Returns:
            float: Trust threshold between 0.0 and 1.0
        """
        if not self.use_dynamic_threshold or round_num < self.window_size:
            # Use static threshold if dynamic not enabled or not enough history
            return self.static_threshold if self.static_threshold is not None else 0.5
            
        # Get recent global scores for dynamic threshold
        recent_scores = []
        for entry in reversed(self.global_trust_history):
            if entry['round'] >= round_num - self.window_size:
                recent_scores.append(entry['score'])
                
        if not recent_scores:
            return self.static_threshold if self.static_threshold is not None else 0.5
            
        # Use percentile as dynamic threshold
        try:
            import numpy as np
            return float(np.percentile(recent_scores, self.percentile_threshold))
        except ImportError:
            # Fallback if numpy not available
            sorted_scores = sorted(recent_scores)
            idx = max(0, int(len(sorted_scores) * self.percentile_threshold / 100) - 1)
            return sorted_scores[idx]
    
    def _detect_anomalies(self, client_id: ClientID, score: float, context: Dict[str, Any]) -> List[str]:
        """
        Detect anomalies in client behavior based on trust scores.
        
        Args:
            client_id: Client identifier
            score: Current trust score
            context: Evaluation context
            
        Returns:
            List[str]: Detected anomalies as flags
        """
        flags = []
        
        # Check for client history
        client_scores = [entry.get('score', 0.0) for entry in self.client_history.get(client_id, [])]
        if len(client_scores) < 2:
            return flags
            
        try:
            import numpy as np
            
            # Check for sudden drop in trust
            if len(client_scores) >= 2:
                prev_score = client_scores[-2]
                drop = prev_score - score
                mean_drop = np.mean([abs(client_scores[i] - client_scores[i+1]) 
                                    for i in range(len(client_scores)-2)])
                std_drop = np.std([abs(client_scores[i] - client_scores[i+1]) 
                                 for i in range(len(client_scores)-2)]) or 0.1
                
                if drop > 0 and drop > mean_drop + self.anomaly_sensitivity * std_drop:
                    flags.append('sudden_trust_drop')
                    
            # Check for oscillating trust pattern
            if len(client_scores) >= 4:
                diffs = [client_scores[i+1] - client_scores[i] for i in range(len(client_scores)-1)]
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                
                if sign_changes >= len(diffs) * 0.75:
                    flags.append('oscillating_trust')
                    
            # Check for consistently low trust
            if len(client_scores) >= 3:
                if all(s < self._get_trust_threshold(context.get('round', 0)) for s in client_scores[-3:]):
                    flags.append('persistent_low_trust')
                    
        except ImportError:
            self.logger.warning("NumPy not available for anomaly detection")
            
        return flags
    
    def update_client_history(self, client_id: ClientID, update_info: Dict[str, Any]) -> None:
        """
        Update client history for trust evaluation.
        
        Args:
            client_id: Client identifier
            update_info: Trust evaluation data
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        
        self.client_history[client_id].append(update_info)
        
        # Limit history size to prevent memory issues
        max_history = self.config.get('max_history_size', 100)
        if len(self.client_history[client_id]) > max_history:
            self.client_history[client_id] = self.client_history[client_id][-max_history:]
    
    def get_client_trust_history(self, client_id: ClientID) -> List[Dict[str, Any]]:
        """
        Get trust history for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            List[Dict[str, Any]]: Trust history entries
        """
        return self.client_history.get(client_id, [])
    
    def get_global_trust_statistics(self) -> Dict[str, Any]:
        """
        Get global trust statistics across all clients.
        
        Returns:
            Dict[str, Any]: Trust statistics
        """
        if not self.global_trust_history:
            return {
                'count': 0,
                'avg_score': 0.0,
                'trusted_rate': 0.0,
            }
            
        try:
            import numpy as np
            scores = [entry['score'] for entry in self.global_trust_history]
            trusted = [entry['trusted'] for entry in self.global_trust_history]
            
            return {
                'count': len(scores),
                'avg_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_dev': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'trusted_rate': float(np.mean([int(t) for t in trusted])),
                'percentiles': {
                    '10': float(np.percentile(scores, 10)),
                    '25': float(np.percentile(scores, 25)),
                    '50': float(np.percentile(scores, 50)),
                    '75': float(np.percentile(scores, 75)),
                    '90': float(np.percentile(scores, 90)),
                }
            }
        except ImportError:
            # Fallback without numpy
            scores = [entry['score'] for entry in self.global_trust_history]
            trusted = [entry['trusted'] for entry in self.global_trust_history]
            
            return {
                'count': len(scores),
                'avg_score': sum(scores) / len(scores),
                'trusted_rate': sum(1 for t in trusted if t) / len(trusted),
                'min_score': min(scores),
                'max_score': max(scores),
            }


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""
    
    @abstractmethod
    def export(self, metrics: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Export metrics to a destination.
        
        Args:
            metrics: List of metrics entries
            output_path: Base path for output
        """
        pass
        
    @abstractmethod
    def log(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log a single metrics entry.
        
        Args:
            metrics: Metrics data to log
            context: Context information
        """
        pass


class CSVMetricsExporter(MetricsExporter):
    """Export metrics to CSV files."""
    
    def export(self, metrics: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            metrics: List of metrics entries
            output_path: Base path for output
        """
        try:
            import csv
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not metrics:
                return
                
            # Get all possible keys from all metrics entries
            fieldnames = set()
            for entry in metrics:
                # Flatten nested dictionaries
                flat_entry = self._flatten_dict(entry)
                fieldnames.update(flat_entry.keys())
            
            # Write to CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                for entry in metrics:
                    # Write flattened entries
                    writer.writerow(self._flatten_dict(entry))
                    
        except ImportError:
            raise RuntimeError("CSV export requires csv module")
            
    def log(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log metrics to in-memory buffer (no direct logging for CSV).
        
        Args:
            metrics: Metrics data to log
            context: Context information
        """
        # CSV exporter doesn't log individual entries - they're batched for export
        pass
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten nested dictionaries for CSV export.
        
        Args:
            d: Dictionary to flatten
            parent_key: Key prefix for nested fields
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}/{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)


class JSONMetricsExporter(MetricsExporter):
    """Export metrics to JSON files."""
    
    def export(self, metrics: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: List of metrics entries
            output_path: Base path for output
        """
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to JSON
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"JSON export failed: {str(e)}")
            
    def log(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Log metrics to in-memory buffer (no direct logging for JSON).
        
        Args:
            metrics: Metrics data to log
            context: Context information
        """
        # JSON exporter doesn't log individual entries - they're batched for export
        pass


# Try to import optional dependencies
try:
    import torch.utils.tensorboard
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class BaseMetrics(MetricsInterface, ABC):
    """
    Base implementation for metrics collection and management.
    
    Features:
    - Multiple backend support (CSV, JSON, TensorBoard, MLflow)
    - Hierarchical metrics organization
    - Automatic flattening of nested metrics
    - Real-time and batch export options
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Exporter configuration
        self.export_format = self.config.get('export_format', 'json')
        self.real_time_export = self.config.get('real_time_export', False)
        self.auto_export_interval = self.config.get('auto_export_interval', 0)  # 0 means no auto export
        self.last_export_time = time.time()
        
        # Pluggable exporters
        self.exporters: Dict[str, MetricsExporter] = {}
        self._register_default_exporters()
        
        # Additional backends
        self._init_tensorboard()
        self._init_mlflow()
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate metrics configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Metrics config must be a dictionary")
            
        # Validate export format
        if self.export_format not in self.exporters and self.export_format != 'all':
            available_formats = list(self.exporters.keys())
            raise ConfigurationError(
                f"Unsupported export format '{self.export_format}'. "
                f"Available formats: {available_formats}"
            )
            
        # Validate auto export interval
        if self.auto_export_interval < 0:
            raise ConfigurationError(f"Auto export interval must be non-negative: {self.auto_export_interval}")
    
    def _register_default_exporters(self) -> None:
        """Register default metrics exporters."""
        self.register_exporter('csv', CSVMetricsExporter())
        self.register_exporter('json', JSONMetricsExporter())
    
    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard exporter if configured."""
        if not self.config.get('use_tensorboard', False):
            return
            
        if not HAS_TENSORBOARD:
            self.logger.warning("TensorBoard requested but torch not available")
            return
            
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Create TensorBoard exporter
            log_dir = self.config.get('tensorboard_log_dir', './runs')
            self.tb_writer = SummaryWriter(log_dir)
            
            # Create and register exporter
            class TensorBoardExporter(MetricsExporter):
                def __init__(self, writer):
                    self.writer = writer
                    
                def export(self, metrics, output_path):
                    # TensorBoard doesn't need batch export - it's logged in real-time
                    pass
                    
                def log(self, metrics, context):
                    # Get step (round number) from context
                    step = context.get('round', 0)
                    
                    # Log each scalar metric
                    flat_metrics = self._flatten_dict(metrics)
                    for key, value in flat_metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(key, value, step)
                            
                def _flatten_dict(self, d, parent_key=''):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{parent_key}/{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.update(self._flatten_dict(v, new_key))
                        else:
                            items[new_key] = v
                    return items
            
            self.register_exporter('tensorboard', TensorBoardExporter(self.tb_writer))
            self.logger.info(f"TensorBoard initialized with log_dir: {log_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard: {str(e)}")
    
    def _init_mlflow(self) -> None:
        """Initialize MLflow exporter if configured."""
        if not self.config.get('use_mlflow', False):
            return
            
        if not HAS_MLFLOW:
            self.logger.warning("MLflow requested but mlflow not available")
            return
            
        try:
            # Set up MLflow tracking URI
            tracking_uri = self.config.get('mlflow_tracking_uri')
            experiment_name = self.config.get('mlflow_experiment', 'trust-mcnet')
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            # Set experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run if auto-start enabled
            if self.config.get('mlflow_auto_start', True):
                run_name = self.config.get('mlflow_run_name')
                mlflow.start_run(run_name=run_name)
                self.logger.info(f"MLflow run started: {run_name or '[default]'}")
                
            # Create and register exporter
            class MLflowExporter(MetricsExporter):
                def export(self, metrics, output_path):
                    # MLflow doesn't need batch export - it's logged in real-time
                    pass
                    
                def log(self, metrics, context):
                    # Flatten metrics for MLflow
                    flat_metrics = self._flatten_dict(metrics)
                    
                    # Log params from context
                    if 'params' in context:
                        mlflow.log_params(context['params'])
                        
                    # Log metrics
                    mlflow_metrics = {k: v for k, v in flat_metrics.items() 
                                     if isinstance(v, (int, float))}
                    mlflow.log_metrics(mlflow_metrics, step=context.get('round', None))
                    
                def _flatten_dict(self, d, parent_key=''):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.update(self._flatten_dict(v, new_key))
                        else:
                            items[new_key] = v
                    return items
            
            self.register_exporter('mlflow', MLflowExporter())
            self.logger.info(f"MLflow initialized with experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {str(e)}")
    
    def register_exporter(self, name: str, exporter: MetricsExporter) -> None:
        """
        Register a custom metrics exporter.
        
        Args:
            name: Unique identifier for this exporter
            exporter: Metrics exporter implementation
        """
        if name in self.exporters:
            self.logger.warning(f"Overwriting existing exporter: {name}")
            
        self.exporters[name] = exporter
        self.logger.info(f"Registered metrics exporter: {name}")
    
    def log_metrics(self, metrics: Metrics, context: Dict[str, Any]) -> None:
        """
        Log metrics with context.
        
        Args:
            metrics: Metrics data
            context: Additional context (e.g., round number)
        """
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
            
        # Add context information if not present
        for key, value in context.items():
            if key not in metrics:
                metrics[key] = value
                
        # Process metrics (implementation-specific)
        processed_metrics = self._process_metrics(metrics, context)
        
        # Save to history
        self.metrics_history.append(processed_metrics)
        
        # Real-time export if enabled
        if self.real_time_export:
            self._export_to_backends(processed_metrics, context)
            
        # Auto-export if interval elapsed
        if self.auto_export_interval > 0:
            current_time = time.time()
            if current_time - self.last_export_time >= self.auto_export_interval:
                self.export_metrics(Path(self.config.get('metrics_dir', './metrics')))
                self.last_export_time = current_time
    
    def _process_metrics(self, metrics: Metrics, context: Dict[str, Any]) -> Metrics:
        """
        Process and transform metrics before logging.
        
        Subclasses can override for custom processing.
        
        Args:
            metrics: Raw metrics data
            context: Additional context
            
        Returns:
            Metrics: Processed metrics
        """
        return metrics
    
    def export_metrics(self, output_path: Path) -> None:
        """
        Export metrics to file(s).
        
        Args:
            output_path: Base path for export
        """
        if not self.metrics_history:
            self.logger.warning("No metrics to export")
            return
            
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to all specified formats
        if self.export_format == 'all':
            for name, exporter in self.exporters.items():
                format_path = output_path / f"metrics_{name}_{timestamp}.{name}"
                try:
                    exporter.export(self.metrics_history, format_path)
                    self.logger.info(f"Exported metrics to {format_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export metrics to {name}: {str(e)}")
        else:
            # Export to specific format
            format_path = output_path / f"metrics_{timestamp}.{self.export_format}"
            try:
                self.exporters[self.export_format].export(self.metrics_history, format_path)
                self.logger.info(f"Exported metrics to {format_path}")
            except Exception as e:
                self.logger.error(f"Failed to export metrics: {str(e)}")
    
    def _export_to_backends(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Export metrics to registered backends in real-time.
        
        Args:
            metrics: Metrics data
            context: Additional context
        """
        for name, exporter in self.exporters.items():
            try:
                exporter.log(metrics, context)
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {name}: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected metrics.
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        if not self.metrics_history:
            return {}
        
        # Basic summary - subclasses can override for specific metrics
        summary = {
            'total_entries': len(self.metrics_history),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'first_timestamp': self.metrics_history[0].get('timestamp') if self.metrics_history else None,
            'last_timestamp': self.metrics_history[-1].get('timestamp') if self.metrics_history else None,
        }
        
        # Add exporters info
        summary['available_exporters'] = list(self.exporters.keys())
        summary['export_format'] = self.export_format
        
        return summary
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics_history.clear()
        self.logger.info("Metrics history cleared")
        
    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        # Close TensorBoard writer if exists
        if hasattr(self, 'tb_writer'):
            try:
                self.tb_writer.close()
            except:
                pass
                
        # End MLflow run if auto-started
        if HAS_MLFLOW and self.config.get('use_mlflow', False) and self.config.get('mlflow_auto_start', True):
            try:
                mlflow.end_run()
            except:
                pass


class BasePartitioner(PartitionerInterface, ABC):
    """
    Base implementation for data partitioners.
    
    Features:
    - Multiple partitioning strategies (IID, non-IID, Dirichlet, etc.)
    - Distribution diagnostics
    - Data quality validation
    - Stratification support
    """
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Partitioning configuration
        self.num_clients = self.config.get('num_clients', 1)
        self.min_samples = self.config.get('min_samples_per_client', 1)
        self.partition_method = self.config.get('method', 'iid')
        self.balance_factor = self.config.get('balance_factor', 1.0)  # 1.0 = balanced
        self.seed = self.config.get('seed', None)
        
        # For non-IID partitioning
        self.alpha = self.config.get('dirichlet_alpha', 1.0)  # Dirichlet concentration parameter
        self.by_label = self.config.get('partition_by_label', False)  # Partition by class labels
        self.custom_splits = self.config.get('custom_splits', None)  # User-defined partition ratios
        
        # Diagnostics
        self.diagnostics_enabled = self.config.get('enable_diagnostics', False)
        
        self._validate_config()
        
        # Set random seed if specified
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def _validate_config(self) -> None:
        """Validate partitioner configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Partitioner config must be a dictionary")
        
        # Validate number of clients
        if self.num_clients < 1:
            raise ConfigurationError(f"num_clients must be >= 1, got {self.num_clients}")
        
        # Validate min_samples
        if self.min_samples < 1:
            raise ConfigurationError(f"min_samples_per_client must be >= 1, got {self.min_samples}")
        
        # Validate balance factor
        if self.balance_factor <= 0:
            raise ConfigurationError(f"balance_factor must be positive, got {self.balance_factor}")
            
        # Validate Dirichlet alpha
        if self.alpha <= 0:
            raise ConfigurationError(f"dirichlet_alpha must be positive, got {self.alpha}")
            
        # Validate custom splits if provided
        if self.custom_splits is not None:
            if len(self.custom_splits) != self.num_clients:
                raise ConfigurationError(
                    f"Length of custom_splits ({len(self.custom_splits)}) "
                    f"must match num_clients ({self.num_clients})"
                )
            if any(s <= 0 for s in self.custom_splits):
                raise ConfigurationError("All custom splits must be positive")
    
    def partition(self, data: Any, num_clients: int = None, **kwargs) -> List[Any]:
        """
        Partition data among clients using selected strategy.
        
        Args:
            data: Data to partition
            num_clients: Override number of clients from config
            **kwargs: Additional partitioning parameters
            
        Returns:
            List[Any]: List of partitions, one per client
        """
        # Use num_clients from parameters if provided, else from config
        if num_clients is not None:
            self.num_clients = num_clients
        
        # Choose partitioning strategy
        method = kwargs.get('method', self.partition_method).lower()
        
        self.logger.info(f"Partitioning data for {self.num_clients} clients using '{method}' method")
        
        try:
            # Call appropriate partitioning method
            if method == 'iid':
                partitions = self._partition_iid(data, **kwargs)
            elif method == 'non-iid' or method == 'noniid':
                if self.by_label:
                    partitions = self._partition_by_label(data, **kwargs)
                else:
                    partitions = self._partition_non_iid(data, **kwargs)
            elif method == 'dirichlet':
                partitions = self._partition_dirichlet(data, **kwargs)
            elif method == 'custom':
                partitions = self._partition_custom(data, **kwargs)
            else:
                raise ConfigurationError(f"Unknown partitioning method: {method}")
                
            # Validate partitioning result
            self.validate_partition(partitions, self.min_samples)
            
            # Generate diagnostics if enabled
            if self.diagnostics_enabled:
                diagnostics = self._generate_diagnostics(data, partitions)
                self.logger.info(f"Partition diagnostics: {diagnostics}")
                
            return partitions
            
        except Exception as e:
            self.logger.error(f"Partitioning failed: {str(e)}")
            raise
    
    def _partition_iid(self, data: Any, **kwargs) -> List[Any]:
        """
        Partition data uniformly (IID) across clients.
        
        Args:
            data: Data to partition
            **kwargs: Additional parameters
            
        Returns:
            List[Any]: Client partitions
        """
        # Implementation-specific IID partitioning
        return self._default_partition(data, self.num_clients)
    
    def _partition_non_iid(self, data: Any, **kwargs) -> List[Any]:
        """
        Partition data non-uniformly (non-IID) across clients.
        
        Args:
            data: Data to partition
            **kwargs: Additional parameters
            
        Returns:
            List[Any]: Client partitions
        """
        # Default implementation - can be overridden by subclasses
        try:
            # Generate skewed partition sizes
            total_size = len(data)
            skew = kwargs.get('skew', 0.5)  # Higher = more skew
            
            # Create a distribution based on power law
            weights = np.power(np.arange(1, self.num_clients + 1), -skew)
            weights = weights / np.sum(weights)
            
            # Calculate samples per client
            samples_per_client = [max(int(total_size * w), self.min_samples) for w in weights]
            
            # Adjust to ensure we don't exceed total samples
            excess = sum(samples_per_client) - total_size
            if excess > 0:
                # Remove excess samples from clients with most samples
                for _ in range(excess):
                    idx = np.argmax(samples_per_client)
                    samples_per_client[idx] -= 1
            elif excess < 0:
                # Add missing samples to clients with fewest samples
                for _ in range(-excess):
                    idx = np.argmin(samples_per_client)
                    samples_per_client[idx] += 1
            
            # Shuffle the data
            indices = list(range(total_size))
            random.shuffle(indices)
            
            # Split indices according to calculated sizes
            partition_indices = []
            start_idx = 0
            for size in samples_per_client:
                partition_indices.append(indices[start_idx:start_idx + size])
                start_idx += size
                
            # Convert indices to actual data partitions
            partitions = []
            for indices in partition_indices:
                partitions.append(self._subset_data(data, indices))
                
            return partitions
            
        except Exception as e:
            self.logger.error(f"Non-IID partitioning failed: {str(e)}")
            # Fall back to IID partitioning
            self.logger.warning("Falling back to IID partitioning")
            return self._partition_iid(data, **kwargs)
    
    def _partition_dirichlet(self, data: Any, **kwargs) -> List[Any]:
        """
        Partition data using Dirichlet distribution (concentration-based non-IID).
        
        Args:
            data: Data to partition
            **kwargs: Additional parameters
            
        Returns:
            List[Any]: Client partitions
        """
        try:
            # Get labels for Dirichlet partitioning
            labels = self._get_data_labels(data)
            if labels is None:
                raise ValueError("Dirichlet partitioning requires labels")
                
            # Get unique labels
            unique_labels = sorted(set(labels))
            num_classes = len(unique_labels)
            
            # Override alpha if provided in kwargs
            alpha = kwargs.get('alpha', self.alpha)
            
            # Generate Dirichlet distribution for each class
            class_distributions = np.random.dirichlet(
                alpha=np.ones(self.num_clients) * alpha, 
                size=num_classes
            )
            
            # Create empty partitions
            client_idxs = [[] for _ in range(self.num_clients)]
            
            # Assign examples of each class to clients according to distribution
            for c, class_idx in enumerate(unique_labels):
                # Find indices of this class
                idxs = np.where(np.array(labels) == class_idx)[0]
                
                # Shuffle indices
                np.random.shuffle(idxs)
                
                # Calculate how many samples of this class go to each client
                proportions = class_distributions[c]
                proportions = proportions / np.sum(proportions)  # Normalize
                class_counts = np.array([int(p * len(idxs)) for p in proportions])
                
                # Adjust to ensure all samples are assigned
                remainder = len(idxs) - np.sum(class_counts)
                if remainder > 0:
                    # Distribute remainder among clients with highest proportions
                    sorted_idx = np.argsort(proportions)[-remainder:]
                    for idx in sorted_idx:
                        class_counts[idx] += 1
                
                # Assign samples to clients
                start_idx = 0
                for client_idx, count in enumerate(class_counts):
                    end_idx = min(start_idx + count, len(idxs))
                    client_idxs[client_idx].extend(idxs[start_idx:end_idx])
                    start_idx = end_idx
            
            # Convert indices to actual data partitions
            partitions = []
            for indices in client_idxs:
                partitions.append(self._subset_data(data, indices))
                
            return partitions
            
        except Exception as e:
            self.logger.error(f"Dirichlet partitioning failed: {str(e)}")
            # Fall back to non-IID partitioning
            self.logger.warning("Falling back to non-IID partitioning")
            return self._partition_non_iid(data, **kwargs)
    
    def _partition_by_label(self, data: Any, **kwargs) -> List[Any]:
        """
        Partition data by assigning different classes to different clients.
        
        Args:
            data: Data to partition
            **kwargs: Additional parameters
            
        Returns:
            List[Any]: Client partitions
        """
        try:
            # Get labels for label-based partitioning
            labels = self._get_data_labels(data)
            if labels is None:
                raise ValueError("Label-based partitioning requires labels")
                
            # Get unique labels
            unique_labels = sorted(set(labels))
            num_classes = len(unique_labels)
            
            if num_classes < self.num_clients:
                self.logger.warning(
                    f"Number of classes ({num_classes}) < number of clients ({self.num_clients}). "
                    "Some classes will be shared."
                )
            
            # Calculate classes per client
            classes_per_client = max(1, num_classes // self.num_clients)
            
            # Shuffle classes
            shuffled_classes = list(unique_labels)
            random.shuffle(shuffled_classes)
            
            # Assign classes to clients
            client_classes = []
            for i in range(self.num_clients):
                start_idx = (i * classes_per_client) % num_classes
                end_idx = ((i + 1) * classes_per_client) % num_classes
                
                if start_idx < end_idx:
                    assigned_classes = shuffled_classes[start_idx:end_idx]
                else:
                    assigned_classes = shuffled_classes[start_idx:] + shuffled_classes[:end_idx]
                
                client_classes.append(assigned_classes)
            
            # Ensure each client has at least one class
            for i, classes in enumerate(client_classes):
                if not classes:
                    # Steal a class from the client with the most classes
                    donor_idx = max(range(len(client_classes)), key=lambda j: len(client_classes[j]))
                    donor_classes = client_classes[donor_idx]
                    if len(donor_classes) > 1:  # Ensure donor has at least one class left
                        donated_class = donor_classes.pop()
                        client_classes[i].append(donated_class)
            
            # Create client partitions based on assigned classes
            partitions = []
            for client_idx, classes in enumerate(client_classes):
                # Get indices for all assigned classes
                client_indices = []
                for c in classes:
                    idxs = np.where(np.array(labels) == c)[0]
                    client_indices.extend(idxs)
                
                # Convert indices to actual data partition
                partitions.append(self._subset_data(data, client_indices))
            
            return partitions
            
        except Exception as e:
            self.logger.error(f"Label-based partitioning failed: {str(e)}")
            # Fall back to Dirichlet partitioning
            self.logger.warning("Falling back to Dirichlet partitioning")
            return self._partition_dirichlet(data, **kwargs)
    
    def _partition_custom(self, data: Any, **kwargs) -> List[Any]:
        """
        Partition data using custom split ratios.
        
        Args:
            data: Data to partition
            **kwargs: Additional parameters including 'splits' for custom ratios
            
        Returns:
            List[Any]: Client partitions
        """
        splits = kwargs.get('splits', self.custom_splits)
        if splits is None or len(splits) != self.num_clients:
            raise ValueError("Custom partitioning requires valid splits for each client")
        
        # Normalize splits
        total = sum(splits)
        normalized_splits = [s / total for s in splits]
        
        # Get total size
        total_size = len(data)
        
        # Calculate samples per client
        samples_per_client = [max(int(total_size * s), self.min_samples) for s in normalized_splits]
        
        # Adjust to ensure we don't exceed total samples
        excess = sum(samples_per_client) - total_size
        if excess > 0:
            # Remove excess samples from clients with most samples
            for _ in range(excess):
                idx = np.argmax(samples_per_client)
                samples_per_client[idx] -= 1
        elif excess < 0:
            # Add missing samples to clients with fewest samples
            for _ in range(-excess):
                idx = np.argmin(samples_per_client)
                samples_per_client[idx] += 1
        
        # Shuffle the data
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Split indices according to calculated sizes
        partition_indices = []
        start_idx = 0
        for size in samples_per_client:
            partition_indices.append(indices[start_idx:start_idx + size])
            start_idx += size
            
        # Convert indices to actual data partitions
        partitions = []
        for indices in partition_indices:
            partitions.append(self._subset_data(data, indices))
            
        return partitions
    
    def validate_partition(self, partitions: List[Any], min_samples_per_client: int = 1) -> None:
        """
        Validate that partitions meet requirements.
        
        Args:
            partitions: List of partitions
            min_samples_per_client: Minimum required samples per client
        """
        if not partitions:
            raise PartitioningError("No partitions created")
        
        if len(partitions) != self.num_clients:
            raise PartitioningError(
                f"Expected {self.num_clients} partitions, got {len(partitions)}"
            )
        
        for i, partition in enumerate(partitions):
            try:
                partition_size = len(partition)
                if partition_size < min_samples_per_client:
                    raise PartitioningError(
                        f"Partition {i} has {partition_size} samples, "
                        f"minimum required: {min_samples_per_client}"
                    )
            except TypeError:
                raise PartitioningError(f"Partition {i} does not support len() operation")
    
    def _default_partition(self, data: Any, num_partitions: int) -> List[Any]:
        """
        Default implementation for partitioning.
        
        Subclasses should override this for data-specific partitioning.
        
        Args:
            data: Data to partition
            num_partitions: Number of partitions to create
            
        Returns:
            List[Any]: List of partitions
        """
        total_size = len(data)
        base_size = total_size // num_partitions
        remainder = total_size % num_partitions
        
        # Shuffle the data
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Distribute indices to partitions
        partitions = []
        start_idx = 0
        for i in range(num_partitions):
            # Add one extra sample for the first 'remainder' partitions
            partition_size = base_size + (1 if i < remainder else 0)
            partition_indices = indices[start_idx:start_idx + partition_size]
            start_idx += partition_size
            
            # Convert indices to actual data partition
            partitions.append(self._subset_data(data, partition_indices))
            
        return partitions
    
    def _subset_data(self, data: Any, indices: List[int]) -> Any:
        """
        Extract a subset of data using indices.
        
        Subclasses should override for data-specific subsetting.
        
        Args:
            data: Full dataset
            indices: Indices to include in subset
            
        Returns:
            Any: Data subset
        """
        # Default implementation - works for list-like objects
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return data[indices]
        except ImportError:
            pass
            
        try:
            # Try common APIs
            if hasattr(data, 'iloc'):  # pandas
                return data.iloc[indices]
            elif hasattr(data, 'subset'):  # some dataset objects
                return data.subset(indices)
            else:
                # Fallback - try to extract elements
                return [data[i] for i in indices]
        except Exception as e:
            raise PartitioningError(f"Failed to subset data: {str(e)}")
    
    def _get_data_labels(self, data: Any) -> List[int]:
        """
        Extract labels from data for class-based partitioning.
        
        Subclasses should override for data-specific label extraction.
        
        Args:
            data: Dataset with labels
            
        Returns:
            List[int]: List of labels
        """
        # Default implementation - try common patterns
        try:
            # Try to access common attributes
            if hasattr(data, 'targets'):  # torchvision, some pytorch datasets
                return data.targets
            elif hasattr(data, 'labels'):  # keras, tensorflow datasets
                return data.labels
            elif hasattr(data, 'y'):  # scikit-learn
                return data.y
            elif isinstance(data, tuple) and len(data) == 2:
                # Assume (features, labels) tuple
                return data[1]
            else:
                self.logger.warning("Could not extract labels from data")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting labels: {str(e)}")
            return None
    
    def _generate_diagnostics(self, data: Any, partitions: List[Any]) -> Dict[str, Any]:
        """
        Generate diagnostics for partitioning.
        
        Args:
            data: Original data
            partitions: Created partitions
            
        Returns:
            Dict[str, Any]: Diagnostics information
        """
        diagnostics = {
            'num_clients': len(partitions),
            'partition_sizes': [len(p) for p in partitions],
            'total_samples': sum(len(p) for p in partitions),
            'original_size': len(data),
        }
        
        # Calculate size statistics
        sizes = diagnostics['partition_sizes']
        diagnostics['min_size'] = min(sizes)
        diagnostics['max_size'] = max(sizes)
        diagnostics['avg_size'] = sum(sizes) / len(sizes)
        diagnostics['size_std_dev'] = (
            sum((s - diagnostics['avg_size']) ** 2 for s in sizes) / len(sizes)
        ) ** 0.5
        
        # Calculate label distribution if possible
        try:
            # Get original labels
            original_labels = self._get_data_labels(data)
            if original_labels is not None:
                original_counts = {}
                for label in original_labels:
                    original_counts[label] = original_counts.get(label, 0) + 1
                
                # Get labels for each partition
                partition_labels = []
                for p in partitions:
                    p_labels = self._get_data_labels(p)
                    if p_labels is None:
                        raise ValueError("Could not extract labels from partition")
                    partition_labels.append(p_labels)
                
                # Calculate label distribution for each partition
                label_distributions = []
                for p_labels in partition_labels:
                    counts = {}
                    for label in p_labels:
                        counts[label] = counts.get(label, 0) + 1
                    label_distributions.append(counts)
                
                diagnostics['label_distributions'] = label_distributions
                diagnostics['original_label_distribution'] = original_counts
                
                # Calculate label diversity (number of unique labels per partition)
                label_diversity = [len(set(p_labels)) for p_labels in partition_labels]
                diagnostics['label_diversity'] = label_diversity
                
        except Exception as e:
            self.logger.warning(f"Could not calculate label distribution: {str(e)}")
            
        return diagnostics


class BaseConfig(ConfigInterface, ABC):
    """
    Base implementation for configuration management.
    
    Features:
    - Loading from multiple sources (YAML, JSON, env vars)
    - Nested configuration access
    - Schema validation
    - Environment variable interpolation
    - Configuration inheritance
    - Default values
    """
    
    def __init__(self, config_data: Union[Dict[str, Any], str, Path]):
        """
        Initialize configuration object.
        
        Args:
            config_data: Dictionary containing configuration data,
                         or path to a configuration file (YAML/JSON)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track configuration sources and overrides
        self.sources: List[str] = []
        self.overrides: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        
        # Configuration data
        self.data: Dict[str, Any] = {}
        self._load_config(config_data)
        
        # Populate with environment variables if enabled
        if self.get('load_env_vars', False):
            self._load_env_vars()
        
        # Validate configuration
        self._validate()
        
        # Apply default values for missing fields
        self._apply_defaults()
    
    def _load_config(self, config_data: Union[Dict[str, Any], str, Path]) -> None:
        """
        Load configuration from source.
        
        Args:
            config_data: Configuration source
        """
        try:
            if isinstance(config_data, dict):
                self.data = config_data
                self.sources.append("dictionary")
            elif isinstance(config_data, (str, Path)):
                path = str(config_data)
                if not os.path.exists(path):
                    raise ConfigurationError(f"Configuration file not found: {path}")
                
                # Detect file type from extension
                if path.endswith('.yaml') or path.endswith('.yml'):
                    self._load_yaml(path)
                    self.sources.append(f"yaml:{path}")
                elif path.endswith('.json'):
                    self._load_json(path)
                    self.sources.append(f"json:{path}")
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {path}"
                    )
            else:
                raise ConfigurationError(
                    f"Unsupported configuration source type: {type(config_data)}"
                )
                
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def _load_yaml(self, path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
        """
        try:
            import yaml
            with open(path, 'r') as f:
                self.data = yaml.safe_load(f) or {}
        except ImportError:
            raise ConfigurationError(
                "PyYAML package required for YAML configuration files"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML file {path}: {str(e)}")
    
    def _load_json(self, path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
        """
        try:
            with open(path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON file {path}: {str(e)}")
    
    def _load_env_vars(self) -> None:
        """
        Load configuration from environment variables.
        
        This will override existing configuration values.
        Environment variables should be prefixed with the configured prefix.
        """
        prefix = self.get('env_var_prefix', 'TRUST_MCNET_')
        
        # Find all environment variables with the prefix
        env_vars = {
            k: v for k, v in os.environ.items() 
            if k.startswith(prefix)
        }
        
        if env_vars:
            self.logger.info(f"Found {len(env_vars)} environment variables with prefix {prefix}")
            
            # Process each environment variable
            for key, value in env_vars.items():
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Replace double underscores with dots for nested keys
                if '__' in config_key:
                    config_key = config_key.replace('__', '.')
                
                # Convert value to appropriate type
                typed_value = self._convert_env_var_value(value)
                
                # Set in configuration
                self.set_nested(config_key, typed_value)
                self.overrides[config_key] = f"env:{key}"
    
    def _convert_env_var_value(self, value: str) -> Any:
        """
        Convert environment variable value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Any: Converted value
        """
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try to parse as boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Load schema if available
        self.schema = self._get_schema()
        
        # If Pydantic is available, use it for schema validation
        if HAS_PYDANTIC and self.schema:
            try:
                from pydantic import BaseModel, Field, create_model
                from pydantic.error_wrappers import ValidationError
                
                # Create dynamic model from schema
                fields = {}
                for key, field_schema in self.schema.items():
                    field_type = field_schema.get('type', Any)
                    required = field_schema.get('required', False)
                    default = field_schema.get('default', ... if required else None)
                    description = field_schema.get('description', '')
                    
                    # Create Field with validation rules
                    fields[key] = (field_type, Field(
                        default=default,
                        description=description
                    ))
                
                # Create and validate model
                ConfigModel = create_model('ConfigModel', **fields)
                ConfigModel(**self.data)
                
            except ValidationError as e:
                errors = "; ".join(str(err) for err in e.errors())
                raise ConfigurationError(f"Configuration validation failed: {errors}")
            except Exception as e:
                self.logger.warning(f"Pydantic schema validation failed: {str(e)}")
                # Fall back to basic validation
                self._basic_validate()
        else:
            # Use basic validation
            self._basic_validate()
    
    def _basic_validate(self) -> None:
        """
        Perform basic validation of configuration.
        
        Subclasses should override for specific validation rules.
        """
        # Check required fields
        required_fields = self._get_required_fields()
        for field in required_fields:
            if not self.has(field):
                raise ConfigurationError(f"Missing required configuration field: {field}")
    
    def _get_required_fields(self) -> List[str]:
        """
        Get list of required configuration fields.
        
        Returns:
            List[str]: List of required field names
        """
        # Extract required fields from schema if available
        if self.schema:
            return [key for key, schema in self.schema.items() if schema.get('required', False)]
        
        # Default implementation - subclasses should override
        return []
    
    def _get_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema.
        
        Returns:
            Dict[str, Any]: Schema definition
        """
        # Default implementation - subclasses should override
        return {}
    
    def _apply_defaults(self) -> None:
        """Apply default values for missing fields."""
        # Apply defaults from schema
        if self.schema:
            for key, field_schema in self.schema.items():
                if 'default' in field_schema and not self.has(key):
                    self.set(key, field_schema['default'])
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.data.get(key, default)
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if path not found
            
        Returns:
            Any: Configuration value
        """
        # Split the key path
        keys = key_path.split('.')
        value = self.data
        
        # Traverse the nested dictionaries
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
            
        return value
    
    def has(self, key: str) -> bool:
        """
        Check if configuration contains key.
        
        Args:
            key: Configuration key
            
        Returns:
            bool: True if key exists
        """
        return key in self.data
    
    def has_nested(self, key_path: str) -> bool:
        """
        Check if nested configuration path exists.
        
        Args:
            key_path: Dot-separated path to configuration value
            
        Returns:
            bool: True if path exists
        """
        # Split the key path
        keys = key_path.split('.')
        value = self.data
        
        # Traverse the nested dictionaries
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
            
        return True
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.data[key] = value
        self.overrides[key] = "runtime"
    
    def set_nested(self, key_path: str, value: Any) -> None:
        """
        Set nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        # Split the key path
        keys = key_path.split('.')
        target = self.data
        
        # Traverse and create nested dictionaries as needed
        for i, key in enumerate(keys[:-1]):
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
            
        # Set the value
        target[keys[-1]] = value
        self.overrides[key_path] = "runtime"
    
    def update(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration with new data.
        
        Args:
            config_data: New configuration data
        """
        self.data.update(config_data)
        for key in config_data:
            self.overrides[key] = "update"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.data.copy()
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert nested configuration to flat dictionary with dot notation keys.
        
        Returns:
            Dict[str, Any]: Flat configuration dictionary
        """
        result = {}
        
        def flatten(prefix: str, config: Dict[str, Any]) -> None:
            for key, value in config.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(path, value)
                else:
                    result[path] = value
                    
        flatten("", self.data)
        return result
    
    def save(self, path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Output path
        """
        # Determine file format from extension
        if path.endswith('.yaml') or path.endswith('.yml'):
            self._save_yaml(path)
        elif path.endswith('.json'):
            self._save_json(path)
        else:
            raise ConfigurationError(f"Unsupported file format: {path}")
    
    def _save_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Output path
        """
        try:
            import yaml
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {path}")
        except ImportError:
            raise ConfigurationError(
                "PyYAML package required for YAML configuration files"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {path}: {str(e)}")
    
    def _save_json(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Output path
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.data, f, indent=2)
            self.logger.info(f"Configuration saved to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {path}: {str(e)}")
            
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Merge with another configuration object.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            BaseConfig: New merged configuration
        """
        # Create a copy of this configuration
        merged_data = self.to_dict()
        
        # Deep merge with other configuration
        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(merged_data, other.to_dict())
        
        # Create new configuration object
        merged = self.__class__(merged_data)
        merged.sources = self.sources + other.sources
        merged.overrides = {**self.overrides, **other.overrides}
        
        return merged


class BaseExperiment(ExperimentInterface, ABC):
    """
    Base implementation for experiments.
    
    Features:
    - Experiment tracking and logging
    - Checkpointing and resumable experiments
    - CLI integration for experiment configuration
    - Results persistence and visualization
    - Resource monitoring and management
    - Event-based experiment lifecycle
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment with configuration.
        
        Args:
            config: Dictionary containing experiment configuration
        """
        self.config = config
        self.experiment_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.end_time = None
        
        # Configure experiment logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.experiment_id}")
        self._setup_logging()
        
        # Experiment state
        self.phase = ExperimentPhase.INITIALIZED
        self.results: Dict[str, Any] = {
            'experiment_id': self.experiment_id,
            'experiment_class': self.__class__.__name__,
            'start_time': None,
            'end_time': None,
            'duration': None,
            'phase': self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase),
        }
        
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: List[str] = []
        self.events: List[Dict[str, Any]] = []
        
        # Experiment progress
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        
        # State tracking
        self.is_setup = False
        self.is_running = False
        self.is_complete = False
        
        # Validate the configuration
        self._validate_config()
        
        # Parse additional parameters from CLI if enabled
        if self.config.get('enable_cli', False):
            self._parse_cli_args()
            
        # Register experiment start event
        self._register_event('experiment_init', {'config': self._sanitize_config()})
    
    def _setup_logging(self) -> None:
        """Configure experiment-specific logging."""
        log_level = self.config.get('log_level', 'INFO').upper()
        log_dir = self.config.get('log_dir', 'logs')
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create experiment log file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(
            log_dir, 
            f"{self.__class__.__name__}_{self.experiment_id}_{timestamp}.log"
        )
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Store log file path in results
        self.results['log_file'] = log_file
        
    def _validate_config(self) -> None:
        """
        Validate experiment configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ConfigurationError("Experiment config must be a dictionary")
        
        # Check for required fields
        required_keys = ['name']
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ConfigurationError(f"Missing required experiment config keys: {missing_keys}")
            
        # Validate specific fields
        if 'max_runtime' in self.config and self.config['max_runtime'] <= 0:
            raise ConfigurationError("max_runtime must be positive")
            
        if 'random_seed' in self.config:
            try:
                seed = int(self.config['random_seed'])
                if seed < 0:
                    raise ConfigurationError("random_seed must be non-negative")
            except (ValueError, TypeError):
                raise ConfigurationError("random_seed must be an integer")
                
        # Extended validation - subclasses should add further validation
        self._validate_extended_config()
    
    def _validate_extended_config(self) -> None:
        """
        Perform extended validation of configuration.
        
        Subclasses should override this method to add custom validation.
        """
        pass
    
    def _sanitize_config(self) -> Dict[str, Any]:
        """
        Sanitize configuration for serialization.
        
        Returns:
            Dict[str, Any]: Sanitized configuration
        """
        return {k: v for k, v in self.config.items() if self._is_serializable(v)}
    
    def _parse_cli_args(self) -> None:
        """Parse command-line arguments to override configuration."""
        import argparse
        parser = argparse.ArgumentParser(description=f"Run {self.__class__.__name__} experiment")
        
        # Add default arguments for common parameters
        parser.add_argument('--name', help='Experiment name')
        parser.add_argument('--results-dir', help='Directory to save results')
        parser.add_argument('--log-level', help='Logging level')
        parser.add_argument('--random-seed', type=int, help='Random seed')
        parser.add_argument('--checkpoint', help='Checkpoint file to resume from')
        parser.add_argument('--max-runtime', type=int, help='Maximum runtime in seconds')
        
        # Add experiment-specific arguments
        self._add_cli_args(parser)
        
        # Parse args and update config
        args = parser.parse_args()
        arg_dict = vars(args)
        
        # Update config with CLI args (if provided)
        for key, value in arg_dict.items():
            if value is not None:
                config_key = key.replace('-', '_')
                self.config[config_key] = value
                self.logger.debug(f"Config override from CLI: {config_key} = {value}")
    
    def _add_cli_args(self, parser: 'argparse.ArgumentParser') -> None:
        """
        Add experiment-specific CLI arguments.
        
        Args:
            parser: Argument parser instance
        """
        # Subclasses should override this method to add custom arguments
        pass
    
    def setup(self) -> None:
        """Set up experiment resources."""
        self.logger.info(f"Setting up experiment: {self.config.get('name')} (ID: {self.experiment_id})")
        self.phase = ExperimentPhase.SETUP
        self.results['phase'] = self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
        
        try:
            # Set random seed if specified
            if 'random_seed' in self.config:
                self._set_random_seed(int(self.config['random_seed']))
            
            # Resume from checkpoint if specified
            if self.config.get('resume_from_checkpoint'):
                checkpoint_path = self.config['resume_from_checkpoint']
                if os.path.exists(checkpoint_path):
                    self._resume_from_checkpoint(checkpoint_path)
                else:
                    self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            
            # Call subclass implementation
            self._setup_experiment()
            
            self.is_setup = True
            self._register_event('experiment_setup', {'success': True})
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}", exc_info=True)
            self._register_event('experiment_setup', {
                'success': False, 
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            raise ExperimentError(f"Setup failed: {str(e)}") from e
    
    @abstractmethod
    def _setup_experiment(self) -> None:
        """
        Implementation-specific setup logic.
        
        Subclasses must override this method.
        """
        pass
    
    def _set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
            
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
            
        self.logger.info(f"Random seed set to {seed}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Dict[str, Any]: Experiment results
        """
        if not self.is_setup:
            self.logger.warning("Experiment not set up. Running setup first.")
            self.setup()
            
        self.logger.info(f"Starting experiment: {self.config.get('name')} (ID: {self.experiment_id})")
        self.phase = ExperimentPhase.RUNNING
        self.results['phase'] = self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
        
        self.start_time = time.time()
        self.results['start_time'] = time.strftime(
            "%Y-%m-%d %H:%M:%S", 
            time.localtime(self.start_time)
        )
        
        self.is_running = True
        self._register_event('experiment_start', {'timestamp': self.results['start_time']})
        
        try:
            # Execute experiment logic
            experiment_results = self._run_experiment()
            
            # Update experiment state
            self.is_complete = True
            self.phase = ExperimentPhase.COMPLETED
            self._register_event('experiment_complete', {'success': True})
            
        except KeyboardInterrupt:
            self.logger.warning("Experiment interrupted by user")
            self.phase = ExperimentPhase.INTERRUPTED
            self._register_event('experiment_interrupt', {'timestamp': time.time()})
            
            # Create checkpoint if enabled
            if self.config.get('enable_checkpointing', False):
                self._create_checkpoint("interrupt")
                
            raise
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            self.phase = ExperimentPhase.FAILED
            self._register_event('experiment_error', {
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            
            # Create checkpoint if enabled
            if self.config.get('enable_checkpointing', False):
                self._create_checkpoint("error")
                
            raise ExperimentError(f"Experiment execution failed: {str(e)}") from e
            
        finally:
            # Record end time and calculate duration
            self.end_time = time.time()
            self.is_running = False
            
            duration = self.end_time - self.start_time
            self.results.update({
                'end_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
                'duration': duration,
                'phase': self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
            })
            
            self.logger.info(f"Experiment completed in {duration:.2f} seconds")
            
            # Save results if experiment was completed
            if self.is_complete and self.config.get('auto_save_results', True):
                self.save_results()
        
        # Merge experiment-specific results with base results
        if experiment_results:
            self.results.update(experiment_results)
            
        return self.get_results()
    
    @abstractmethod
    def _run_experiment(self) -> Optional[Dict[str, Any]]:
        """
        Implementation-specific experiment execution logic.
        
        Subclasses must override this method.
        
        Returns:
            Optional[Dict[str, Any]]: Experiment-specific results, if any
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up experiment resources."""
        self.logger.info(f"Cleaning up experiment: {self.config.get('name')} (ID: {self.experiment_id})")
        self.phase = ExperimentPhase.CLEANUP
        self.results['phase'] = self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
        
        try:
            # Call implementation-specific cleanup
            self._cleanup_experiment()
            self._register_event('experiment_cleanup', {'success': True})
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
            self._register_event('experiment_cleanup', {
                'success': False,
                'error': str(e),
                'error_type': e.__class__.__name__
            })
    
    @abstractmethod
    def _cleanup_experiment(self) -> None:
        """
        Implementation-specific cleanup logic.
        
        Subclasses must override this method.
        """
        pass
    
    def get_phase(self) -> ExperimentPhase:
        """Get current experiment phase."""
        return self.phase
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results."""
        # Update phase in case it changed
        self.results['phase'] = self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
        return self.results.copy()
    
    def log_result(self, key: str, value: Any) -> None:
        """
        Log a result value.
        
        Args:
            key: Result key
            value: Result value
        """
        self.results[key] = value
        self.logger.info(f"Logged result: {key} = {value}")
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a specific step.
        
        Args:
            step: Step number
            metrics: Metrics dictionary
        """
        if step not in self.metrics:
            self.metrics[step] = {}
            
        # Update metrics for this step
        self.metrics[step].update(metrics)
        
        # Log to console
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        self.logger.info(f"Step {step} metrics: {metrics_str}")
        
        # Update results with latest metrics
        self.results['latest_metrics'] = self.metrics[step].copy()
        self.results['latest_step'] = step
        
        # Register metrics event
        self._register_event('metrics_update', {
            'step': step,
            'metrics': metrics
        })
        
        # Create checkpoint if scheduled
        if self.config.get('enable_checkpointing', False):
            checkpoint_interval = self.config.get('checkpoint_interval', 0)
            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                self._create_checkpoint("scheduled")
    
    def save_results(self, results_dir: str = None) -> str:
        """
        Save experiment results to disk.
        
        Args:
            results_dir: Directory to save results to
            
        Returns:
            str: Path to saved results file
        """
        results_dir = results_dir or self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Update experiment metadata
        self.results.update({
            'experiment_id': self.experiment_id,
            'experiment_class': self.__class__.__name__,
            'config': self._sanitize_config(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': self.metrics,
            'is_complete': self.is_complete,
            'phase': self.phase.name if isinstance(self.phase, ExperimentPhase) else str(self.phase)
        })
        
        # Get git info if available
        if HAS_GIT:
            try:
                repo = git.Repo(search_parent_directories=True)
                self.results['git_info'] = {
                    'commit': repo.head.commit.hexsha,
                    'branch': repo.active_branch.name,
                    'dirty': repo.is_dirty()
                }
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                pass
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(
            results_dir, 
            f"{self.__class__.__name__}_{self.experiment_id}_{timestamp}.json"
        )
        
        # Save results
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=self._json_serializer)
                
            self.logger.info(f"Results saved to {filename}")
            self._register_event('results_saved', {'path': filename})
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            self._register_event('results_save_error', {
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            return None
    
    def _is_serializable(self, obj: Any) -> bool:
        """
        Check if an object can be serialized to JSON.
        
        Args:
            obj: Object to check
            
        Returns:
            bool: True if object can be serialized
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for complex objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Any: JSON serializable representation
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            return str(obj)
    
    def _create_checkpoint(self, reason: str = "scheduled") -> Optional[str]:
        """
        Create experiment checkpoint.
        
        Args:
            reason: Reason for checkpoint creation
            
        Returns:
            Optional[str]: Path to checkpoint file or None if failed
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint data
        checkpoint_data = {
            'experiment_id': self.experiment_id,
            'experiment_class': self.__class__.__name__,
            'config': self._sanitize_config(),
            'results': self.results,
            'metrics': self.metrics,
            'step': self.current_step,
            'timestamp': time.time(),
            'reason': reason,
            'is_complete': self.is_complete,
            'state': self._get_checkpoint_state()
        }
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(
            checkpoint_dir, 
            f"{self.__class__.__name__}_{self.experiment_id}_{timestamp}.ckpt"
        )
        
        # Save checkpoint
        try:
            with open(filename, 'wb') as f:
                import pickle
                pickle.dump(checkpoint_data, f)
                
            self.checkpoints.append(filename)
            self.logger.info(f"Checkpoint created at {filename}")
            
            self._register_event('checkpoint_created', {
                'path': filename,
                'reason': reason
            })
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {str(e)}")
            self._register_event('checkpoint_error', {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'reason': reason
            })
            return None
    
    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get additional state to include in checkpoint.
        
        Subclasses should override to save additional state.
        
        Returns:
            Dict[str, Any]: Additional state to checkpoint
        """
        return {}
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Resume experiment from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: True if resumption was successful
        """
        self.logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            with open(checkpoint_path, 'rb') as f:
                import pickle
                checkpoint_data = pickle.load(f)
                
            # Verify checkpoint compatibility
            if checkpoint_data['experiment_class'] != self.__class__.__name__:
                self.logger.warning(
                    f"Checkpoint experiment class ({checkpoint_data['experiment_class']}) "
                    f"doesn't match current class ({self.__class__.__name__})"
                )
                
            # Restore experiment state
            self.experiment_id = checkpoint_data['experiment_id']
            self.results = checkpoint_data['results']
            self.metrics = checkpoint_data['metrics']
            self.current_step = checkpoint_data['step']
            
            # Restore additional state
            self._restore_checkpoint_state(checkpoint_data['state'])
            
            self.logger.info(
                f"Successfully resumed from checkpoint (step {self.current_step})"
            )
            
            self._register_event('checkpoint_resumed', {
                'path': checkpoint_path,
                'step': self.current_step
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {str(e)}")
            self._register_event('checkpoint_resume_error', {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'path': checkpoint_path
            })
            return False
    
    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """
        Restore additional state from checkpoint.
        
        Subclasses should override to restore additional state.
        
        Args:
            state: State to restore
        """
        pass
        
    def _register_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Register experiment event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data or {}
        }
        self.events.append(event)
        
    def visualize_results(self, viz_type: str = 'metrics') -> Optional[str]:
        """
        Visualize experiment results.
        
        Args:
            viz_type: Type of visualization to create
            
        Returns:
            Optional[str]: Path to visualization file or None if failed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for visualization")
            return None
            
        if not self.metrics:
            self.logger.warning("No metrics available to visualize")
            return None
            
        # Create visualization
        if viz_type == 'metrics':
            return self._visualize_metrics()
        elif viz_type == 'summary':
            return self._visualize_summary()
        else:
            self.logger.warning(f"Unknown visualization type: {viz_type}")
            return None
            
    def _visualize_metrics(self) -> Optional[str]:
        """
        Create visualization of metrics over time.
        
        Returns:
            Optional[str]: Path to visualization file
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get steps and metrics
            steps = sorted(int(s) for s in self.metrics.keys())
            metric_names = set()
            for step_metrics in self.metrics.values():
                metric_names.update(step_metrics.keys())
                
            # Create figure
            num_metrics = len(metric_names)
            fig, axes = plt.subplots(
                num_metrics, 1, 
                figsize=(10, 3 * num_metrics),
                sharex=True
            )
            
            # Adjust for single metric case
            if num_metrics == 1:
                axes = [axes]
                
            # Plot each metric
            for i, metric_name in enumerate(sorted(metric_names)):
                values = []
                for step in steps:
                    step_key = str(step)
                    if step_key in self.metrics and metric_name in self.metrics[step_key]:
                        values.append(self.metrics[step_key][metric_name])
                    else:
                        values.append(None)  # Use None for missing values
                
                # Filter out None values
                valid_points = [(s, v) for s, v in zip(steps, values) if v is not None]
                if valid_points:
                    valid_steps, valid_values = zip(*valid_points)
                    axes[i].plot(valid_steps, valid_values, 'o-', label=metric_name)
                    axes[i].set_title(metric_name)
                    axes[i].grid(True)
                    axes[i].set_ylabel(metric_name)
            
            # Set labels on bottom plot
            axes[-1].set_xlabel('Step')
            
            # Add title
            plt.suptitle(f"Experiment Metrics: {self.config.get('name', 'Unnamed')}")
            fig.tight_layout()
            
            # Save figure
            results_dir = self.config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(
                results_dir, 
                f"{self.__class__.__name__}_{self.experiment_id}_{timestamp}_metrics.png"
            )
            
            fig.savefig(filename)
            plt.close(fig)
            
            self.logger.info(f"Metrics visualization saved to {filename}")
            self._register_event('visualization_created', {
                'type': 'metrics',
                'path': filename
            })
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to visualize metrics: {str(e)}")
            self._register_event('visualization_error', {
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            return None
    
    def _visualize_summary(self) -> Optional[str]:
        """
        Create summary visualization of experiment results.
        
        Returns:
            Optional[str]: Path to visualization file
        """
        # Subclasses should override this method
        return None
        
    def should_stop_early(self) -> bool:
        """
        Check if experiment should stop early.
        
        Returns:
            bool: True if experiment should stop
        """
        # Check max runtime
        if 'max_runtime' in self.config and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.config['max_runtime']:
                self.logger.info(f"Maximum runtime reached: {elapsed:.2f}/{self.config['max_runtime']} seconds")
                return True
                
        # Check for early stopping based on metrics
        if self._check_early_stopping_criteria():
            self.logger.info("Early stopping criteria met")
            return True
            
        return False
    
    def _check_early_stopping_criteria(self) -> bool:
        """
        Check if early stopping criteria are met.
        
        Subclasses should override to implement specific criteria.
        
        Returns:
            bool: True if early stopping criteria are met
        """
        return False
