"""
Metrics logging system for TRUST-MCNet federated learning.

This module provides comprehensive logging capabilities including:
- TensorBoard integration
- MLflow support (optional)
- CSV metrics export
- Real-time monitoring
"""

import logging
import time
import csv
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class MetricsLogger(ABC):
    """Abstract base class for metrics logging."""
    
    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        pass
    
    @abstractmethod
    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dictionary of metrics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger and cleanup resources."""
        pass


class TensorBoardLogger(MetricsLogger):
    """TensorBoard metrics logger."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard logger initialized: {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, using CSV fallback")
            self.writer = None
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log scalar metric to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
            self.writer.flush()
    
    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log dictionary of metrics to TensorBoard."""
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, step)
            self.writer.flush()
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class CSVLogger(MetricsLogger):
    """CSV metrics logger."""
    
    def __init__(self, log_file: str):
        """
        Initialize CSV logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.fieldnames = ['step', 'timestamp', 'metric_name', 'value']
        self.file_handle = open(self.log_file, 'w', newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        logger.info(f"CSV logger initialized: {self.log_file}")
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log scalar metric to CSV."""
        self.writer.writerow({
            'step': step,
            'timestamp': time.time(),
            'metric_name': name,
            'value': value
        })
        self.file_handle.flush()
    
    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log dictionary of metrics to CSV."""
        timestamp = time.time()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.writerow({
                    'step': step,
                    'timestamp': timestamp,
                    'metric_name': name,
                    'value': value
                })
        self.file_handle.flush()
    
    def close(self) -> None:
        """Close CSV file."""
        self.file_handle.close()


class JSONLogger(MetricsLogger):
    """JSON metrics logger."""
    
    def __init__(self, log_file: str):
        """
        Initialize JSON logger.
        
        Args:
            log_file: Path to JSON log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_data = []
        
        logger.info(f"JSON logger initialized: {self.log_file}")
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log scalar metric to JSON."""
        self.metrics_data.append({
            'step': step,
            'timestamp': time.time(),
            'metric_name': name,
            'value': value
        })
    
    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log dictionary of metrics to JSON."""
        timestamp = time.time()
        for name, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                self.metrics_data.append({
                    'step': step,
                    'timestamp': timestamp,
                    'metric_name': name,
                    'value': value
                })
    
    def close(self) -> None:
        """Save JSON data to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)


class CompositeLogger(MetricsLogger):
    """Composite logger that writes to multiple backends."""
    
    def __init__(self, loggers: List[MetricsLogger]):
        """
        Initialize composite logger.
        
        Args:
            loggers: List of metrics loggers
        """
        self.loggers = loggers
        logger.info(f"Composite logger initialized with {len(loggers)} backends")
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log scalar to all backends."""
        for logger_backend in self.loggers:
            try:
                logger_backend.log_scalar(name, value, step)
            except Exception as e:
                logger.warning(f"Failed to log scalar to backend: {e}")
    
    def log_dict(self, metrics: Dict[str, Any], step: int) -> None:
        """Log dictionary to all backends."""
        for logger_backend in self.loggers:
            try:
                logger_backend.log_dict(metrics, step)
            except Exception as e:
                logger.warning(f"Failed to log dict to backend: {e}")
    
    def close(self) -> None:
        """Close all backends."""
        for logger_backend in self.loggers:
            try:
                logger_backend.close()
            except Exception as e:
                logger.warning(f"Failed to close backend: {e}")


class FederatedMetricsManager:
    """Manager for federated learning metrics."""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str = "federated_experiment"):
        """
        Initialize federated metrics manager.
        
        Args:
            config: Logging configuration
            experiment_name: Name of the experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        self.current_round = 0
        
        # Initialize loggers based on configuration
        self.loggers = self._setup_loggers()
        
        # Metrics storage
        self.round_metrics = []
        self.client_metrics = {}
        
        logger.info(f"Federated metrics manager initialized for experiment: {experiment_name}")
    
    def _setup_loggers(self) -> List[MetricsLogger]:
        """Setup logging backends based on configuration."""
        loggers = []
        
        # Always include CSV logger
        csv_path = f"logs/{self.experiment_name}/metrics.csv"
        loggers.append(CSVLogger(csv_path))
        
        # Add TensorBoard if enabled
        if self.config.get('metrics', {}).get('enable_tensorboard', False):
            tb_dir = self.config['metrics'].get('tensorboard_dir', 'logs/tensorboard')
            tb_path = f"{tb_dir}/{self.experiment_name}"
            loggers.append(TensorBoardLogger(tb_path))
        
        # Add JSON logger for structured data
        json_path = f"logs/{self.experiment_name}/metrics.json"
        loggers.append(JSONLogger(json_path))
        
        return loggers
    
    def start_round(self, round_num: int) -> None:
        """Start logging for a new federated round."""
        self.current_round = round_num
        logger.info(f"Starting metrics logging for round {round_num}")
    
    def log_server_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log server-side metrics."""
        try:
            # Add round and timestamp information
            enriched_metrics = {
                **metrics,
                'round': self.current_round,
                'timestamp': time.time(),
                'type': 'server'
            }
            
            # Log to all backends
            for metrics_logger in self.loggers:
                metrics_logger.log_dict(enriched_metrics, self.current_round)
            
            # Store for analysis
            self.round_metrics.append(enriched_metrics)
            
            logger.debug(f"Logged server metrics for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Failed to log server metrics: {e}")
    
    def log_client_metrics(self, client_id: str, metrics: Dict[str, Any]) -> None:
        """Log client-specific metrics."""
        try:
            # Add client and round information
            enriched_metrics = {
                **metrics,
                'client_id': client_id,
                'round': self.current_round,
                'timestamp': time.time(),
                'type': 'client'
            }
            
            # Create client-specific metric names
            client_specific_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    client_specific_metrics[f"client_{client_id}_{key}"] = value
            
            # Log to all backends
            for metrics_logger in self.loggers:
                metrics_logger.log_dict(client_specific_metrics, self.current_round)
            
            # Store for analysis
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = []
            self.client_metrics[client_id].append(enriched_metrics)
            
            logger.debug(f"Logged client metrics for {client_id}, round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Failed to log client metrics for {client_id}: {e}")
    
    def log_aggregation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log model aggregation metrics."""
        try:
            # Add aggregation-specific information
            enriched_metrics = {
                **metrics,
                'round': self.current_round,
                'timestamp': time.time(),
                'type': 'aggregation'
            }
            
            # Log to all backends
            for metrics_logger in self.loggers:
                metrics_logger.log_dict(enriched_metrics, self.current_round)
            
            logger.debug(f"Logged aggregation metrics for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Failed to log aggregation metrics: {e}")
    
    def log_trust_metrics(self, trust_scores: Dict[str, float]) -> None:
        """Log trust evaluation metrics."""
        try:
            # Create trust-specific metrics
            trust_metrics = {
                f"trust_score_{client_id}": score 
                for client_id, score in trust_scores.items()
            }
            
            trust_metrics.update({
                'avg_trust_score': sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0,
                'min_trust_score': min(trust_scores.values()) if trust_scores else 0,
                'max_trust_score': max(trust_scores.values()) if trust_scores else 0,
                'round': self.current_round,
                'timestamp': time.time(),
                'type': 'trust'
            })
            
            # Log to all backends
            for metrics_logger in self.loggers:
                metrics_logger.log_dict(trust_metrics, self.current_round)
            
            logger.debug(f"Logged trust metrics for round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Failed to log trust metrics: {e}")
    
    def get_round_summary(self) -> Dict[str, Any]:
        """Get summary of metrics for the current round."""
        try:
            round_data = [m for m in self.round_metrics if m.get('round') == self.current_round]
            
            if not round_data:
                return {}
            
            # Calculate summary statistics
            summary = {
                'round': self.current_round,
                'num_metrics': len(round_data),
                'timestamp': time.time()
            }
            
            # Add aggregated metrics if available
            numeric_metrics = {}
            for metric_data in round_data:
                for key, value in metric_data.items():
                    if isinstance(value, (int, float)) and key not in ['round', 'timestamp']:
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(value)
            
            # Calculate statistics
            for key, values in numeric_metrics.items():
                if values:
                    summary[f'{key}_mean'] = sum(values) / len(values)
                    summary[f'{key}_min'] = min(values)
                    summary[f'{key}_max'] = max(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate round summary: {e}")
            return {}
    
    def export_metrics(self, export_path: str) -> None:
        """Export all metrics to a comprehensive report."""
        try:
            export_data = {
                'experiment_name': self.experiment_name,
                'export_timestamp': time.time(),
                'total_rounds': self.current_round,
                'round_metrics': self.round_metrics,
                'client_metrics': self.client_metrics
            }
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to: {export_file}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def close(self) -> None:
        """Close all logging backends."""
        try:
            for metrics_logger in self.loggers:
                metrics_logger.close()
            
            logger.info(f"Metrics manager closed for experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Error closing metrics manager: {e}")


def create_metrics_manager(config: Dict[str, Any], experiment_name: str = None) -> FederatedMetricsManager:
    """
    Factory function to create metrics manager.
    
    Args:
        config: Configuration dictionary
        experiment_name: Optional experiment name
        
    Returns:
        Configured FederatedMetricsManager instance
    """
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    return FederatedMetricsManager(config, experiment_name)
