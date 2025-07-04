"""
Refactored metrics collection and logging module using interface-based architecture.

This module implements metrics collection that conforms to the MetricsInterface,
providing:
- Registry pattern for metrics collectors
- Interface-based design for extensibility
- Production-grade error handling
- Comprehensive metrics tracking
- Scalable metrics storage and analysis
"""

import logging
import json
import csv
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

from core.interfaces import MetricsInterface
from core.abstractions import BaseMetrics
from core.exceptions import MetricsError, ConfigurationError
from core.types import ConfigType, MetricsData, MetricsSnapshot

logger = logging.getLogger(__name__)


class FederatedLearningMetrics(BaseMetrics):
    """
    Comprehensive metrics collector for federated learning experiments.
    
    Tracks training metrics, client performance, trust scores,
    resource usage, and aggregation statistics.
    """
    
    def __init__(self, config: ConfigType, **kwargs) -> None:
        """
        Initialize federated learning metrics collector.
        
        Args:
            config: Metrics configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        
        self.experiment_name = config.get('experiment_name', f'experiment_{int(datetime.now().timestamp())}')
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.save_format = config.get('save_format', ['json', 'csv'])
        self.save_frequency = config.get('save_frequency', 'round')  # 'round', 'epoch', 'batch'
        
        # Metrics storage
        self.round_metrics = []
        self.client_metrics = defaultdict(list)
        self.trust_metrics = defaultdict(list)
        self.aggregation_metrics = []
        self.resource_metrics = []
        self.global_metrics = {}
        
        # Performance tracking
        self.best_metrics = {}
        self.convergence_tracker = deque(maxlen=config.get('convergence_window', 5))
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FL metrics collector for experiment: {self.experiment_name}")
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int, Dict[str, Any]],
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Record a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/round number
            metadata: Additional metadata
            **kwargs: Additional parameters
        """
        try:
            timestamp = datetime.now().isoformat()
            
            metric_entry = {
                'name': name,
                'value': value,
                'step': step,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }
            
            # Categorize metrics
            if name.startswith('client_'):
                client_id = metadata.get('client_id') if metadata else 'unknown'
                self.client_metrics[client_id].append(metric_entry)
            elif name.startswith('trust_'):
                self.trust_metrics[name].append(metric_entry)
            elif name.startswith('resource_'):
                self.resource_metrics.append(metric_entry)
            elif name.startswith('aggregation_'):
                self.aggregation_metrics.append(metric_entry)
            else:
                self.global_metrics[name] = metric_entry
            
            # Update best metrics
            if isinstance(value, (int, float)):
                if name.endswith('_accuracy') or name.endswith('_f1') or name.endswith('_precision'):
                    # Higher is better
                    if name not in self.best_metrics or value > self.best_metrics[name]['value']:
                        self.best_metrics[name] = metric_entry.copy()
                elif name.endswith('_loss') or name.endswith('_error'):
                    # Lower is better
                    if name not in self.best_metrics or value < self.best_metrics[name]['value']:
                        self.best_metrics[name] = metric_entry.copy()
            
            logger.debug(f"Recorded metric {name}: {value}")
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
            raise MetricsError(f"Failed to record metric: {e}") from e
    
    def record_batch(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Record a batch of metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step/round number
            metadata: Additional metadata
            **kwargs: Additional parameters
        """
        try:
            for name, value in metrics.items():
                self.record_metric(name, value, step, metadata, **kwargs)
            
            # Save if configured
            if self.save_frequency == 'batch':
                self.save()
                
        except Exception as e:
            logger.error(f"Error recording metric batch: {e}")
            raise MetricsError(f"Failed to record metric batch: {e}") from e
    
    def record_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        **kwargs
    ) -> None:
        """
        Record metrics for a complete federated learning round.
        
        Args:
            round_num: Round number
            metrics: Round metrics
            **kwargs: Additional parameters
        """
        try:
            round_entry = {
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics.copy(),
                'metadata': kwargs
            }
            
            self.round_metrics.append(round_entry)
            
            # Check for convergence
            if 'loss' in metrics:
                self.convergence_tracker.append(metrics['loss'])
            
            # Update global metrics
            for name, value in metrics.items():
                self.record_metric(f'round_{name}', value, round_num)
            
            # Save if configured
            if self.save_frequency == 'round':
                self.save()
            
            logger.info(f"Recorded metrics for round {round_num}")
            
        except Exception as e:
            logger.error(f"Error recording round metrics: {e}")
            raise MetricsError(f"Failed to record round metrics: {e}") from e
    
    def record_client_metrics(
        self,
        client_id: str,
        metrics: Dict[str, Any],
        round_num: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Record metrics for a specific client.
        
        Args:
            client_id: Client identifier
            metrics: Client metrics
            round_num: Round number
            **kwargs: Additional parameters
        """
        try:
            client_entry = {
                'client_id': client_id,
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics.copy(),
                'metadata': kwargs
            }
            
            self.client_metrics[client_id].append(client_entry)
            
            # Record individual metrics with client prefix
            for name, value in metrics.items():
                self.record_metric(
                    f'client_{name}',
                    value,
                    round_num,
                    {'client_id': client_id, **kwargs}
                )
            
            logger.debug(f"Recorded metrics for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error recording client metrics: {e}")
            raise MetricsError(f"Failed to record client metrics: {e}") from e
    
    def get_summary(self, include_history: bool = False) -> MetricsSnapshot:
        """
        Get a summary of all collected metrics.
        
        Args:
            include_history: Whether to include full history
            
        Returns:
            Metrics summary
        """
        try:
            summary = {
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'total_rounds': len(self.round_metrics),
                'total_clients': len(self.client_metrics),
                'best_metrics': self.best_metrics.copy(),
                'convergence_status': self._check_convergence()
            }
            
            # Add latest round metrics
            if self.round_metrics:
                latest_round = self.round_metrics[-1]
                summary['latest_round'] = latest_round['round']
                summary['latest_metrics'] = latest_round['metrics']
            
            # Add client summary statistics
            if self.client_metrics:
                summary['client_summary'] = self._compute_client_summary()
            
            # Add trust summary
            if self.trust_metrics:
                summary['trust_summary'] = self._compute_trust_summary()
            
            # Add resource summary
            if self.resource_metrics:
                summary['resource_summary'] = self._compute_resource_summary()
            
            # Include full history if requested
            if include_history:
                summary['history'] = {
                    'round_metrics': self.round_metrics,
                    'client_metrics': dict(self.client_metrics),
                    'trust_metrics': dict(self.trust_metrics),
                    'aggregation_metrics': self.aggregation_metrics,
                    'resource_metrics': self.resource_metrics
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            raise MetricsError(f"Failed to generate summary: {e}") from e
    
    def save(self, filename: Optional[str] = None) -> None:
        """
        Save metrics to disk.
        
        Args:
            filename: Optional custom filename
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.experiment_name}_{timestamp}"
            
            summary = self.get_summary(include_history=True)
            
            # Save in requested formats
            for format_type in self.save_format:
                if format_type == 'json':
                    self._save_json(summary, filename)
                elif format_type == 'csv':
                    self._save_csv(summary, filename)
                else:
                    logger.warning(f"Unknown save format: {format_type}")
            
            logger.info(f"Saved metrics to {self.output_dir}/{filename}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise MetricsError(f"Failed to save metrics: {e}") from e
    
    def _save_json(self, summary: Dict[str, Any], filename: str) -> None:
        """Save metrics as JSON."""
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _save_csv(self, summary: Dict[str, Any], filename: str) -> None:
        """Save metrics as CSV files."""
        # Save round metrics
        if summary.get('history', {}).get('round_metrics'):
            csv_path = self.output_dir / f"{filename}_rounds.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                headers = ['round', 'timestamp']
                if summary['history']['round_metrics']:
                    sample_metrics = summary['history']['round_metrics'][0]['metrics']
                    headers.extend(sample_metrics.keys())
                writer.writerow(headers)
                
                # Data
                for round_data in summary['history']['round_metrics']:
                    row = [round_data['round'], round_data['timestamp']]
                    row.extend([round_data['metrics'].get(h, '') for h in headers[2:]])
                    writer.writerow(row)
        
        # Save client metrics
        if summary.get('history', {}).get('client_metrics'):
            csv_path = self.output_dir / f"{filename}_clients.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                headers = ['client_id', 'round', 'timestamp']
                all_metrics = set()
                for client_data in summary['history']['client_metrics'].values():
                    for entry in client_data:
                        all_metrics.update(entry['metrics'].keys())
                headers.extend(sorted(all_metrics))
                writer.writerow(headers)
                
                # Data
                for client_id, client_data in summary['history']['client_metrics'].items():
                    for entry in client_data:
                        row = [client_id, entry['round'], entry['timestamp']]
                        row.extend([entry['metrics'].get(h, '') for h in headers[3:]])
                        writer.writerow(row)
    
    def _compute_client_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for client metrics."""
        summary = {
            'total_clients': len(self.client_metrics),
            'avg_metrics_per_client': np.mean([len(data) for data in self.client_metrics.values()]),
        }
        
        # Aggregate client performance
        all_accuracies = []
        all_losses = []
        
        for client_data in self.client_metrics.values():
            for entry in client_data:
                metrics = entry['metrics']
                if 'accuracy' in metrics:
                    all_accuracies.append(metrics['accuracy'])
                if 'loss' in metrics:
                    all_losses.append(metrics['loss'])
        
        if all_accuracies:
            summary['accuracy_stats'] = {
                'mean': np.mean(all_accuracies),
                'std': np.std(all_accuracies),
                'min': np.min(all_accuracies),
                'max': np.max(all_accuracies)
            }
        
        if all_losses:
            summary['loss_stats'] = {
                'mean': np.mean(all_losses),
                'std': np.std(all_losses),
                'min': np.min(all_losses),
                'max': np.max(all_losses)
            }
        
        return summary
    
    def _compute_trust_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for trust metrics."""
        summary = {}
        
        for trust_metric, entries in self.trust_metrics.items():
            values = [entry['value'] for entry in entries if isinstance(entry['value'], (int, float))]
            if values:
                summary[trust_metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
    
    def _compute_resource_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for resource metrics."""
        summary = {}
        
        resource_types = defaultdict(list)
        for entry in self.resource_metrics:
            resource_types[entry['name']].append(entry['value'])
        
        for resource_type, values in resource_types.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                summary[resource_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values) if resource_type.endswith('_time') else None
                }
        
        return summary
    
    def _check_convergence(self) -> Dict[str, Any]:
        """Check convergence status based on recent loss values."""
        if len(self.convergence_tracker) < 3:
            return {'status': 'insufficient_data', 'trend': None}
        
        recent_losses = list(self.convergence_tracker)
        
        # Check if loss is decreasing
        trends = []
        for i in range(1, len(recent_losses)):
            trends.append(recent_losses[i] - recent_losses[i-1])
        
        avg_trend = np.mean(trends)
        
        if avg_trend < -0.001:
            status = 'converging'
        elif avg_trend > 0.001:
            status = 'diverging'
        else:
            status = 'stable'
        
        return {
            'status': status,
            'trend': avg_trend,
            'recent_losses': recent_losses,
            'variance': np.var(recent_losses)
        }


# Metrics Registry
class MetricsRegistry:
    """Registry for metrics collectors."""
    
    _metrics = {
        'federated_learning': FederatedLearningMetrics,
    }
    
    @classmethod
    def register(cls, name: str, metrics_class: type) -> None:
        """Register a new metrics collector."""
        if not issubclass(metrics_class, MetricsInterface):
            raise ValueError(f"Metrics collector {metrics_class} must implement MetricsInterface")
        cls._metrics[name] = metrics_class
        logger.info(f"Registered metrics collector: {name}")
    
    @classmethod
    def create(cls, name: str, config: ConfigType, **kwargs) -> MetricsInterface:
        """Create a metrics collector instance."""
        if name not in cls._metrics:
            available = ', '.join(cls._metrics.keys())
            raise MetricsError(f"Metrics collector '{name}' not found. Available: {available}")
        
        metrics_class = cls._metrics[name]
        return metrics_class(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available metrics collectors."""
        return list(cls._metrics.keys())


def create_metrics_collector(name: str, config: ConfigType, **kwargs) -> MetricsInterface:
    """
    Factory function to create metrics collector instances.
    
    Args:
        name: Metrics collector name
        config: Metrics configuration
        **kwargs: Additional parameters
        
    Returns:
        Metrics collector instance
    """
    return MetricsRegistry.create(name, config, **kwargs)


# Export public interface
__all__ = [
    'FederatedLearningMetrics',
    'MetricsRegistry',
    'create_metrics_collector'
]
