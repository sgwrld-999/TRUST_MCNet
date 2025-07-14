"""
Interfaces for TRUST-MCNet federated learning framework.

This module defines the contracts that all components must implement,
following the Interface Segregation Principle (ISP) and Dependency
Inversion Principle (DIP) from SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, runtime_checkable
import torch
from torch.utils.data import Dataset
import numpy as np
from omegaconf import DictConfig


@runtime_checkable
class DataLoaderInterface(Protocol):
    """Interface for data loading and preprocessing components."""
    
    @abstractmethod
    def load_data(self, config: DictConfig) -> Tuple[Dataset, Dataset]:
        """Load training and test datasets.
        
        Args:
            config: Configuration object containing data loading parameters
            
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Raises:
            DataLoadingError: If data cannot be loaded or processed
        """
        ...
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary containing data shape, number of classes, etc.
        """
        ...
    
    @abstractmethod
    def validate_data(self, dataset: Dataset) -> bool:
        """Validate that the dataset meets requirements.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if valid, False otherwise
        """
        ...


@runtime_checkable
class ModelInterface(Protocol):
    """Interface for neural network models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        ...
    
    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays.
        
        Returns:
            List of parameter arrays
        """
        ...
    
    @abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays.
        
        Args:
            parameters: List of parameter arrays
        """
        ...
    
    @abstractmethod
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary containing training metrics
        """
        ...
    
    @abstractmethod
    def eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one evaluation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        ...


@runtime_checkable
class StrategyInterface(Protocol):
    """Interface for federated learning strategies."""
    
    @abstractmethod
    def configure_fit(self, server_round: int, parameters: List[np.ndarray], 
                     client_manager: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure the fit process for a round.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Client manager instance
            
        Returns:
            List of (client, fit_config) tuples
        """
        ...
    
    @abstractmethod
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], 
                     failures: List[Any]) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate training results.
        
        Args:
            server_round: Current round number
            results: List of successful training results
            failures: List of failed training attempts
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        ...
    
    @abstractmethod
    def configure_evaluate(self, server_round: int, parameters: List[np.ndarray],
                          client_manager: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure the evaluation process for a round.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Client manager instance
            
        Returns:
            List of (client, eval_config) tuples
        """
        ...


@runtime_checkable
class TrustEvaluatorInterface(Protocol):
    """Interface for trust evaluation mechanisms."""
    
    @abstractmethod
    def evaluate_trust(self, client_id: str, metrics: Dict[str, float]) -> float:
        """Evaluate trust score for a single client.
        
        Args:
            client_id: Unique identifier for the client
            metrics: Client performance metrics
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        ...
    
    @abstractmethod
    def evaluate_trust_batch(self, client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Evaluate trust scores for multiple clients.
        
        Args:
            client_metrics: Dictionary mapping client_id to metrics
            
        Returns:
            Dictionary mapping client_id to trust score
        """
        ...
    
    @abstractmethod
    def update_trust_history(self, client_id: str, trust_score: float, 
                           round_number: int) -> None:
        """Update trust history for a client.
        
        Args:
            client_id: Unique identifier for the client
            trust_score: Trust score for this round
            round_number: Current round number
        """
        ...


@runtime_checkable
class MetricsInterface(Protocol):
    """Interface for metrics collection and logging."""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics for a given step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        ...
    
    @abstractmethod
    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve logged metrics.
        
        Args:
            metric_names: Optional list of specific metrics to retrieve
            
        Returns:
            Dictionary of metrics
        """
        ...
    
    @abstractmethod
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        ...


@runtime_checkable
class PartitionerInterface(Protocol):
    """Interface for data partitioning strategies."""
    
    @abstractmethod
    def partition(self, dataset: Dataset, num_clients: int, 
                 **kwargs) -> List[Dataset]:
        """Partition dataset among clients.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            **kwargs: Additional partitioning parameters
            
        Returns:
            List of client datasets
        """
        ...
    
    @abstractmethod
    def validate_partition(self, partitions: List[Dataset], 
                          min_samples: int = 1) -> bool:
        """Validate that partitions meet requirements.
        
        Args:
            partitions: List of partitioned datasets
            min_samples: Minimum samples per partition
            
        Returns:
            True if valid, False otherwise
        """
        ...


@runtime_checkable
class ConfigInterface(Protocol):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> DictConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration object
        """
        ...
    
    @abstractmethod
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        ...
    
    @abstractmethod
    def merge_configs(self, *configs: DictConfig) -> DictConfig:
        """Merge multiple configurations.
        
        Args:
            *configs: Variable number of configuration objects
            
        Returns:
            Merged configuration
        """
        ...


@runtime_checkable
class ExperimentInterface(Protocol):
    """Interface for experiment management."""
    
    @abstractmethod
    def setup_experiment(self, config: DictConfig) -> None:
        """Set up the experiment environment.
        
        Args:
            config: Experiment configuration
        """
        ...
    
    @abstractmethod
    def run_experiment(self) -> Dict[str, Any]:
        """Run the federated learning experiment.
        
        Returns:
            Experiment results
        """
        ...
    
    @abstractmethod
    def cleanup_experiment(self) -> None:
        """Clean up experiment resources."""
        ...
    
    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save experiment results.
        
        Args:
            results: Experiment results to save
            output_path: Path to save results
        """
        ...
