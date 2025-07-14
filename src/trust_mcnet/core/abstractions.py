"""
Base abstract classes implementing core interfaces for TRUST-MCNet.

This module provides default implementations and common functionality
that concrete classes can inherit from, reducing code duplication
and ensuring consistent behavior across the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

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
    DatasetInfo,
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
    
    Provides validation, logging, and error handling that all data loaders need.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Config must be a dictionary")
        
        required_keys = ['name']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required config key: {key}")
    
    @abstractmethod
    def load_data(self) -> Tuple[Any, Any]:
        """Load training and test datasets."""
        pass
    
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
    
    Provides parameter handling, validation, and utility methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
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
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: ModelParameters) -> None:
        """Set model parameters."""
        pass
    
    @abstractmethod
    def train(self, data: Any) -> Metrics:
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, data: Any) -> Metrics:
        """Evaluate the model."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'type': self.config.get('type'),
            'config': self.config
        }


class BaseStrategy(StrategyInterface, ABC):
    """
    Base implementation for federated learning strategies.
    
    Provides common aggregation logic and client management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate strategy configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Strategy config must be a dictionary")
        
        required_keys = ['name']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required strategy config key: {key}")
    
    @abstractmethod
    def aggregate(self, client_updates: List[Tuple[ClientID, ModelParameters, Metrics]]) -> ModelParameters:
        """Aggregate client updates."""
        pass
    
    @abstractmethod
    def select_clients(self, available_clients: List[ClientID]) -> List[ClientID]:
        """Select clients for the next round."""
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy."""
        return {
            'name': self.config.get('name'),
            'config': self.config
        }


class BaseTrustEvaluator(TrustEvaluatorInterface, ABC):
    """
    Base implementation for trust evaluators.
    
    Provides common trust calculation and validation logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client_history: Dict[ClientID, List[Dict[str, Any]]] = {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate trust evaluator configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Trust evaluator config must be a dictionary")
        
        # Validate threshold if present
        threshold = self.config.get('threshold')
        if threshold is not None:
            if not (0.0 <= threshold <= 1.0):
                raise ConfigurationError(f"Trust threshold must be between 0.0 and 1.0, got {threshold}")
    
    @abstractmethod
    def evaluate_trust(self, client_id: ClientID, model_update: ModelParameters, 
                      context: Dict[str, Any]) -> TrustScore:
        """Evaluate trust score for a client."""
        pass
    
    def update_client_history(self, client_id: ClientID, update_info: Dict[str, Any]) -> None:
        """Update client history for trust evaluation."""
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        
        self.client_history[client_id].append(update_info)
        
        # Limit history size to prevent memory issues
        max_history = self.config.get('max_history_size', 100)
        if len(self.client_history[client_id]) > max_history:
            self.client_history[client_id] = self.client_history[client_id][-max_history:]
    
    def get_client_trust_history(self, client_id: ClientID) -> List[Dict[str, Any]]:
        """Get trust history for a client."""
        return self.client_history.get(client_id, [])


class BaseMetrics(MetricsInterface, ABC):
    """
    Base implementation for metrics collection and management.
    
    Provides common metrics handling and export functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history: List[Dict[str, Any]] = []
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate metrics configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Metrics config must be a dictionary")
    
    @abstractmethod
    def log_metrics(self, metrics: Metrics, context: Dict[str, Any]) -> None:
        """Log metrics with context."""
        pass
    
    @abstractmethod
    def export_metrics(self, output_path: Path) -> None:
        """Export metrics to file."""
        pass
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        if not self.metrics_history:
            return {}
        
        # Basic summary - subclasses can override for specific metrics
        return {
            'total_entries': len(self.metrics_history),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else None
        }
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics_history.clear()
        self.logger.info("Metrics history cleared")


class BasePartitioner(PartitionerInterface, ABC):
    """
    Base implementation for data partitioners.
    
    Provides common validation and utility methods for data partitioning.
    """
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate partitioner configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Partitioner config must be a dictionary")
        
        num_clients = self.config.get('num_clients', 0)
        if num_clients < 1:
            raise ConfigurationError(f"num_clients must be >= 1, got {num_clients}")
    
    @abstractmethod
    def partition(self, data: Any, num_clients: int, **kwargs) -> List[Any]:
        """Partition data among clients."""
        pass
    
    def validate_partition(self, partitions: List[Any], min_samples_per_client: int = 1) -> None:
        """Validate that partitions meet requirements."""
        if not partitions:
            raise PartitioningError("No partitions created")
        
        if len(partitions) != self.config.get('num_clients'):
            raise PartitioningError(
                f"Expected {self.config.get('num_clients')} partitions, got {len(partitions)}"
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


class BaseConfig(ConfigInterface, ABC):
    """
    Base implementation for configuration management.
    
    Provides validation, loading, and utility methods for configurations.
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        self.data = config_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate()
    
    @abstractmethod
    def _validate(self) -> None:
        """Validate the configuration."""
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.data.copy()


class BaseExperiment(ExperimentInterface, ABC):
    """
    Base implementation for experiments.
    
    Provides common experiment lifecycle management and logging.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.phase = ExperimentPhase.INITIALIZED
        self.results: Dict[str, Any] = {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate experiment configuration."""
        if not isinstance(self.config, dict):
            raise ConfigurationError("Experiment config must be a dictionary")
        
        required_keys = ['name']
        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required experiment config key: {key}")
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the experiment."""
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup after experiment."""
        pass
    
    def get_phase(self) -> ExperimentPhase:
        """Get current experiment phase."""
        return self.phase
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results."""
        return self.results.copy()
    
    def log_result(self, key: str, value: Any) -> None:
        """Log an experiment result."""
        self.results[key] = value
        self.logger.info(f"Logged result: {key} = {value}")
