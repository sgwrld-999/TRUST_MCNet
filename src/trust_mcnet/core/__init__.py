"""
Core module for TRUST-MCNet federated learning framework.

This module contains the fundamental abstractions and interfaces that define
the architecture of the TRUST-MCNet system following SOLID principles.
"""

__version__ = "2.0.0"
__author__ = "TRUST-MCNet Team"

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

from .abstractions import (
    BaseDataLoader,
    BaseModel,
    BaseStrategy,
    BaseTrustEvaluator,
    BaseMetrics,
    BasePartitioner,
    BaseConfig,
    BaseExperiment
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

from .types import (
    ClientID,
    ModelParameters,
    Metrics,
    TrustScore,
    ClientConfig,
    ExperimentConfig
)

__all__ = [
    # Interfaces
    "DataLoaderInterface",
    "ModelInterface", 
    "StrategyInterface",
    "TrustEvaluatorInterface",
    "MetricsInterface",
    "PartitionerInterface",
    "ConfigInterface",
    "ExperimentInterface",
    
    # Base classes
    "BaseDataLoader",
    "BaseModel",
    "BaseStrategy", 
    "BaseTrustEvaluator",
    "BaseMetrics",
    "BasePartitioner",
    "BaseConfig",
    "BaseExperiment",
    
    # Exceptions
    "TrustMCNetError",
    "ConfigurationError",
    "DataLoadingError",
    "ModelError",
    "TrustEvaluationError",
    "PartitioningError",
    "ExperimentError",
    
    # Types
    "ClientID",
    "ModelParameters",
    "Metrics",
    "TrustScore",
    "ClientConfig",
    "ExperimentConfig"
]
