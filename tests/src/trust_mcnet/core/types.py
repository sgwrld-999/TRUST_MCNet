"""
Type definitions for TRUST-MCNet federated learning framework.

This module defines custom types and type aliases to improve code clarity,
type safety, and maintainability throughout the application.
"""

from typing import TypeVar, Dict, List, Any, Union, Tuple, NewType
from enum import Enum, auto
import numpy as np

# Type aliases for better code readability
ConfigType = Dict[str, Any]
"""Configuration dictionary type."""

ClientID = NewType("ClientID", str)
"""Unique identifier for a federated learning client."""

RoundNumber = NewType("RoundNumber", int)
"""Federated learning round number."""

TrustScore = NewType("TrustScore", float)
"""Trust score value between 0.0 and 1.0."""

ModelParameters = List[np.ndarray]
"""Model parameters as a list of numpy arrays."""

Metrics = Dict[str, Union[float, int, str]]
"""Dictionary containing various metrics."""

ClientMetrics = Dict[ClientID, Metrics]
"""Dictionary mapping client IDs to their metrics."""

TrustScores = Dict[ClientID, TrustScore]
"""Dictionary mapping client IDs to their trust scores."""

# Configuration types
DatasetConfig = Dict[str, Any]
"""Configuration dictionary for dataset parameters."""

ModelConfig = Dict[str, Any]
"""Configuration dictionary for model parameters."""

FederatedConfig = Dict[str, Any]
"""Configuration dictionary for federated learning parameters."""

TrustConfig = Dict[str, Any]
"""Configuration dictionary for trust evaluation parameters."""

ExperimentConfig = Dict[str, Any]
"""Configuration dictionary for experiment parameters."""

ClientConfig = Dict[str, Any]
"""Configuration dictionary for client parameters."""

PartitionConfig = Dict[str, Any]
"""Configuration dictionary for data partitioning parameters."""

# Data information type
DatasetInfo = Dict[str, Any]
"""Dictionary containing dataset information like shape, size, classes, etc."""

# Generic types
T = TypeVar('T')
"""Generic type variable."""

Dataset = TypeVar('Dataset')
"""Generic dataset type."""

Model = TypeVar('Model')
"""Generic model type."""


class ExperimentPhase(Enum):
    """Enumeration of experiment phases."""
    INITIALIZED = auto()
    SETTING_UP = auto()
    DATA_LOADING = auto()
    MODEL_SETUP = auto()
    CLIENT_SETUP = auto()
    READY = auto()
    RUNNING = auto()
    TRAINING = auto()
    EVALUATION = auto()
    TRUST_EVALUATION = auto()
    AGGREGATION = auto()
    COMPLETED = auto()
    FAILED = auto()
    CLEANED_UP = auto()


class ClientStatus(Enum):
    """Enumeration of client statuses."""
    IDLE = auto()
    TRAINING = auto()
    EVALUATING = auto()
    UPLOADING = auto()
    FAILED = auto()
    DISCONNECTED = auto()
    TRUSTED = auto()
    UNTRUSTED = auto()


class TrustLevel(Enum):
    """Enumeration of trust levels."""
    VERY_LOW = (0.0, 0.2)
    LOW = (0.2, 0.4)
    MEDIUM = (0.4, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)
    
    def __init__(self, min_score: float, max_score: float):
        self.min_score = min_score
        self.max_score = max_score
    
    @classmethod
    def from_score(cls, score: float) -> 'TrustLevel':
        """Determine trust level from score."""
        for level in cls:
            if level.min_score <= score <= level.max_score:
                return level
        raise ValueError(f"Invalid trust score: {score}")


class DataDistribution(Enum):
    """Enumeration of data distribution strategies."""
    IID = "iid"
    NON_IID_DIRICHLET = "dirichlet"
    NON_IID_PATHOLOGICAL = "pathological"
    NON_IID_PRACTICAL = "practical"


class AggregationStrategy(Enum):
    """Enumeration of aggregation strategies."""
    FEDAVG = "fedavg"
    FEDADAM = "fedadam"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    TRUST_WEIGHTED = "trust_weighted"


class ModelArchitecture(Enum):
    """Enumeration of supported model architectures."""
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RESNET = "resnet"


class OptimizationMetric(Enum):
    """Enumeration of optimization metrics."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"


class ScalingStrategy(Enum):
    """Enumeration of scaling strategies for large experiments."""
    HORIZONTAL = "horizontal"  # More clients
    VERTICAL = "vertical"      # More powerful clients
    HIERARCHICAL = "hierarchical"  # Multiple levels
    DYNAMIC = "dynamic"        # Adaptive scaling


# Result types
TrainingResult = Tuple[ModelParameters, int, Metrics]
"""Training result containing parameters, number of samples, and metrics."""

EvaluationResult = Tuple[int, float, Metrics]
"""Evaluation result containing number of samples, loss, and metrics."""

ExperimentResult = Dict[str, Any]
"""Complete experiment result dictionary."""

# Callback types
ProgressCallback = callable
"""Callback function for progress updates."""

ValidationCallback = callable
"""Callback function for validation."""

# Resource types
ComputeResources = Dict[str, Union[int, float]]
"""Dictionary containing compute resource specifications."""

NetworkResources = Dict[str, Union[int, float]]
"""Dictionary containing network resource specifications."""

StorageResources = Dict[str, Union[int, float]]
"""Dictionary containing storage resource specifications."""

# Utility types
PathLike = Union[str, bytes]
"""Path-like object."""

JSONSerializable = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
"""JSON serializable types."""

ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
"""Valid configuration value types."""

# Advanced types for production features
class SecurityLevel(Enum):
    """Security levels for production deployment."""
    BASIC = auto()
    STANDARD = auto()
    HIGH = auto()
    CRITICAL = auto()


class DeploymentMode(Enum):
    """Deployment modes for different environments."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Factory types
ClientFactory = callable
"""Factory function for creating clients."""

ModelFactory = callable
"""Factory function for creating models."""

StrategyFactory = callable
"""Factory function for creating strategies."""

DataLoaderFactory = callable
"""Factory function for creating data loaders."""
