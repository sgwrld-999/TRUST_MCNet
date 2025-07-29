"""
Metrics collection utility for TRUST-MCNet.

This module provides tools for collecting, storing, and analyzing
system and trust metrics from the federated learning process.
"""

import time
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class TrustMetricsCollector:
    """
    Collects and manages metrics for TRUST-MCNet components.
    
    Features:
    - Time-series metrics storage
    - Aggregation and statistical analysis
    - Persistence to disk
    - Memory-efficient storage with rolling windows
    """
    
    def __init__(self, max_history: int = 10000, storage_path: str = "metrics"):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics points to store in memory
            storage_path: Path for storing metrics files
        """
        self.max_history = max_history
        self.storage_path = storage_path
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Metrics storage
        self.system_metrics = []
        self.trust_metrics = []
        self.training_metrics = []
        self.client_metrics = {}  # Dict[client_id, List[metrics]]
        
        # Track metric timestamps for efficient filtering
        self.system_timestamps = []
        self.trust_timestamps = []
        self.training_timestamps = []
        
        logger.info(f"Metrics collector initialized with max_history={max_history}")
    
    def add_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add system metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        # Ensure timestamp exists
        if 'timestamp' not in metrics:
            metrics['timestamp'] = time.time()
            
        # Add to storage with rolling window
        self.system_metrics.append(metrics)
        self.system_timestamps.append(metrics['timestamp'])
        
        # Enforce max history
        if len(self.system_metrics) > self.max_history:
            self.system_metrics.pop(0)
            self.system_timestamps.pop(0)
    
    def add_trust_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add trust-related metrics.
        
        Args:
            metrics: Dictionary of trust metrics
        """
        # Ensure timestamp exists
        if 'timestamp' not in metrics:
            metrics['timestamp'] = time.time()
            
        # Add to storage with rolling window
        self.trust_metrics.append(metrics)
        self.trust_timestamps.append(metrics['timestamp'])
        
        # Enforce max history
        if len(self.trust_metrics) > self.max_history:
            self.trust_metrics.pop(0)
            self.trust_timestamps.pop(0)
    
    def add_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add training-related metrics.
        
        Args:
            metrics: Dictionary of training metrics
        """
        # Ensure timestamp exists
        if 'timestamp' not in metrics:
            metrics['timestamp'] = time.time()
            
        # Add to storage with rolling window
        self.training_metrics.append(metrics)
        self.training_timestamps.append(metrics['timestamp'])
        
        # Enforce max history
        if len(self.training_metrics) > self.max_history:
            self.training_metrics.pop(0)
            self.training_timestamps.pop(0)
    
    def add_client_metrics(self, client_id: str, metrics: Dict[str, Any]) -> None:
        """
        Add client-specific metrics.
        
        Args:
            client_id: Unique client identifier
            metrics: Dictionary of client metrics
        """
        # Ensure timestamp exists
        if 'timestamp' not in metrics:
            metrics['timestamp'] = time.time()
            
        # Initialize client metrics list if needed
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
            
        # Add metrics with rolling window
        self.client_metrics[client_id].append(metrics)
        
        # Enforce max history
        if len(self.client_metrics[client_id]) > self.max_history:
            self.client_metrics[client_id].pop(0)
    
    def get_system_metrics(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get system metrics within a time range.
        
        Args:
            start_time: Start timestamp (None for all history)
            end_time: End timestamp (None for up to present)
            
        Returns:
            List of system metrics dictionaries
        """
        if not self.system_metrics:
            return []
            
        if start_time is None and end_time is None:
            # Return all metrics (copy to avoid modification)
            return self.system_metrics.copy()
        
        # Use binary search for efficient filtering
        start_idx = 0
        end_idx = len(self.system_metrics)
        
        if start_time is not None:
            start_idx = self._binary_search_timestamp(self.system_timestamps, start_time)
            
        if end_time is not None:
            end_idx = self._binary_search_timestamp(self.system_timestamps, end_time)
            # Include the entry at end_idx if it's within the time range
            if end_idx < len(self.system_timestamps) and self.system_timestamps[end_idx] <= end_time:
                end_idx += 1
                
        return self.system_metrics[start_idx:end_idx]
    
    def get_trust_metrics(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trust metrics within a time range.
        
        Args:
            start_time: Start timestamp (None for all history)
            end_time: End timestamp (None for up to present)
            
        Returns:
            List of trust metrics dictionaries
        """
        if not self.trust_metrics:
            return []
            
        if start_time is None and end_time is None:
            # Return all metrics (copy to avoid modification)
            return self.trust_metrics.copy()
        
        # Use binary search for efficient filtering
        start_idx = 0
        end_idx = len(self.trust_metrics)
        
        if start_time is not None:
            start_idx = self._binary_search_timestamp(self.trust_timestamps, start_time)
            
        if end_time is not None:
            end_idx = self._binary_search_timestamp(self.trust_timestamps, end_time)
            # Include the entry at end_idx if it's within the time range
            if end_idx < len(self.trust_timestamps) and self.trust_timestamps[end_idx] <= end_time:
                end_idx += 1
                
        return self.trust_metrics[start_idx:end_idx]
    
    def get_training_metrics(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training metrics within a time range.
        
        Args:
            start_time: Start timestamp (None for all history)
            end_time: End timestamp (None for up to present)
            
        Returns:
            List of training metrics dictionaries
        """
        if not self.training_metrics:
            return []
            
        if start_time is None and end_time is None:
            # Return all metrics (copy to avoid modification)
            return self.training_metrics.copy()
        
        # Use binary search for efficient filtering
        start_idx = 0
        end_idx = len(self.training_metrics)
        
        if start_time is not None:
            start_idx = self._binary_search_timestamp(self.training_timestamps, start_time)
            
        if end_time is not None:
            end_idx = self._binary_search_timestamp(self.training_timestamps, end_time)
            # Include the entry at end_idx if it's within the time range
            if end_idx < len(self.training_timestamps) and self.training_timestamps[end_idx] <= end_time:
                end_idx += 1
                
        return self.training_metrics[start_idx:end_idx]
    
    def get_client_metrics(
        self,
        client_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get client-specific metrics within a time range.
        
        Args:
            client_id: Unique client identifier
            start_time: Start timestamp (None for all history)
            end_time: End timestamp (None for up to present)
            
        Returns:
            List of client metrics dictionaries
        """
        if client_id not in self.client_metrics or not self.client_metrics[client_id]:
            return []
            
        client_data = self.client_metrics[client_id]
        
        if start_time is None and end_time is None:
            # Return all metrics (copy to avoid modification)
            return client_data.copy()
        
        # Filter by time range
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = float('inf')
            
        return [
            metrics for metrics in client_data
            if start_time <= metrics['timestamp'] <= end_time
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'timestamp': time.time(),
            'system_metrics_count': len(self.system_metrics),
            'trust_metrics_count': len(self.trust_metrics),
            'training_metrics_count': len(self.training_metrics),
            'client_count': len(self.client_metrics),
            'total_client_metrics': sum(len(metrics) for metrics in self.client_metrics.values())
        }
        
        # Add latest metrics if available
        if self.system_metrics:
            summary['latest_system_metrics'] = self.system_metrics[-1]
        if self.trust_metrics:
            summary['latest_trust_metrics'] = self.trust_metrics[-1]
        if self.training_metrics:
            summary['latest_training_metrics'] = self.training_metrics[-1]
            
        return summary
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save metrics to disk.
        
        Args:
            filename: Custom filename (default: auto-generated from timestamp)
            
        Returns:
            Path to the saved metrics file
        """
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_{timestamp}.json"
            
        filepath = os.path.join(self.storage_path, filename)
        
        # Prepare data for serialization
        metrics_data = {
            'timestamp': time.time(),
            'system_metrics': self.system_metrics,
            'trust_metrics': self.trust_metrics,
            'training_metrics': self.training_metrics,
            'client_metrics': self.client_metrics
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Saved metrics to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {str(e)}")
            raise
    
    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load metrics from a file.
        
        Args:
            filepath: Path to the metrics file
            
        Returns:
            Loaded metrics dictionary
        """
        try:
            with open(filepath, 'r') as f:
                metrics_data = json.load(f)
                
            # Update current metrics with loaded data
            if 'system_metrics' in metrics_data:
                self.system_metrics = metrics_data['system_metrics']
                self.system_timestamps = [m.get('timestamp', 0) for m in self.system_metrics]
                
            if 'trust_metrics' in metrics_data:
                self.trust_metrics = metrics_data['trust_metrics']
                self.trust_timestamps = [m.get('timestamp', 0) for m in self.trust_metrics]
                
            if 'training_metrics' in metrics_data:
                self.training_metrics = metrics_data['training_metrics']
                self.training_timestamps = [m.get('timestamp', 0) for m in self.training_metrics]
                
            if 'client_metrics' in metrics_data:
                self.client_metrics = metrics_data['client_metrics']
                
            logger.info(f"Loaded metrics from {filepath}")
            return metrics_data
        except Exception as e:
            logger.error(f"Error loading metrics from {filepath}: {str(e)}")
            raise
    
    def _binary_search_timestamp(self, timestamps: List[float], target: float) -> int:
        """
        Binary search to find the first index where timestamp >= target.
        
        Args:
            timestamps: Sorted list of timestamps
            target: Target timestamp
            
        Returns:
            Index of the first timestamp >= target (or len(timestamps) if none found)
        """
        left, right = 0, len(timestamps)
        
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < target:
                left = mid + 1
            else:
                right = mid
                
        return left


class TrustMetricsAnalyzer:
    """
    Advanced analysis tools for trust metrics.
    
    Features:
    - Trend analysis
    - Anomaly detection
    - Correlation analysis
    - Visualization helpers
    """
    
    def __init__(self, metrics_collector: TrustMetricsCollector):
        """
        Initialize metrics analyzer.
        
        Args:
            metrics_collector: TrustMetricsCollector instance
        """
        self.metrics_collector = metrics_collector
    
    def analyze_trust_trends(
        self,
        window_size: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze trends in trust metrics.
        
        Args:
            window_size: Window size for moving averages
            start_time: Start timestamp for analysis
            end_time: End timestamp for analysis
            
        Returns:
            Dictionary containing trend analysis results
        """
        # Get trust metrics in range
        metrics = self.metrics_collector.get_trust_metrics(start_time, end_time)
        
        if not metrics or len(metrics) < window_size:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract trust scores and timestamps
        trust_scores = []
        timestamps = []
        
        for m in metrics:
            if 'mean_trust' in m:
                trust_scores.append(m['mean_trust'])
                timestamps.append(m['timestamp'])
        
        if not trust_scores:
            return {'error': 'No trust score data available'}
            
        # Convert to numpy arrays
        trust_array = np.array(trust_scores)
        time_array = np.array(timestamps)
        
        # Calculate moving averages
        ma = self._moving_average(trust_array, window_size)
        
        # Detect trends (positive slope = improving, negative = declining)
        if len(ma) >= 2:
            slope = np.polyfit(np.arange(len(ma)), ma, 1)[0]
            trend = 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable')
        else:
            slope = 0
            trend = 'unknown'
            
        # Calculate basic statistics
        stats = {
            'mean': float(np.mean(trust_array)),
            'std': float(np.std(trust_array)),
            'min': float(np.min(trust_array)),
            'max': float(np.max(trust_array))
        }
        
        # Detect significant changes
        significant_changes = self._detect_significant_changes(trust_array, window_size)
        
        return {
            'trend': trend,
            'slope': float(slope),
            'statistics': stats,
            'moving_average': ma.tolist(),
            'significant_changes': significant_changes,
            'data_points': len(trust_array)
        }
    
    def detect_trust_anomalies(
        self,
        threshold_std: float = 2.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies in trust metrics.
        
        Args:
            threshold_std: Number of standard deviations to consider anomalous
            start_time: Start timestamp for analysis
            end_time: End timestamp for analysis
            
        Returns:
            Dictionary containing anomaly detection results
        """
        # Get trust metrics in range
        metrics = self.metrics_collector.get_trust_metrics(start_time, end_time)
        
        if not metrics:
            return {'error': 'No data available for anomaly detection'}
            
        # Extract trust scores and timestamps
        trust_scores = []
        timestamps = []
        
        for m in metrics:
            if 'mean_trust' in m:
                trust_scores.append(m['mean_trust'])
                timestamps.append(m['timestamp'])
                
        if not trust_scores:
            return {'error': 'No trust score data available'}
            
        # Convert to numpy arrays
        trust_array = np.array(trust_scores)
        time_array = np.array(timestamps)
        
        # Calculate statistics
        mean = np.mean(trust_array)
        std = np.std(trust_array)
        threshold_high = mean + threshold_std * std
        threshold_low = mean - threshold_std * std
        
        # Find anomalies
        high_anomalies = np.where(trust_array > threshold_high)[0]
        low_anomalies = np.where(trust_array < threshold_low)[0]
        
        # Prepare result
        result = {
            'anomaly_threshold_std': threshold_std,
            'mean': float(mean),
            'std': float(std),
            'threshold_high': float(threshold_high),
            'threshold_low': float(threshold_low),
            'high_anomalies': [
                {'index': int(i), 'timestamp': float(time_array[i]), 'value': float(trust_array[i])}
                for i in high_anomalies
            ],
            'low_anomalies': [
                {'index': int(i), 'timestamp': float(time_array[i]), 'value': float(trust_array[i])}
                for i in low_anomalies
            ],
            'total_anomalies': len(high_anomalies) + len(low_anomalies),
            'anomaly_rate': (len(high_anomalies) + len(low_anomalies)) / len(trust_array) if len(trust_array) > 0 else 0
        }
        
        return result
    
    def analyze_client_performance(self, client_id: str) -> Dict[str, Any]:
        """
        Analyze a specific client's performance metrics.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Dictionary containing client performance analysis
        """
        # Get client metrics
        metrics = self.metrics_collector.get_client_metrics(client_id)
        
        if not metrics:
            return {'error': f'No metrics available for client {client_id}'}
            
        # Extract relevant metrics
        timestamps = []
        trust_scores = []
        accuracies = []
        training_times = []
        
        for m in metrics:
            timestamps.append(m.get('timestamp', 0))
            
            if 'trust_score' in m:
                trust_scores.append(m['trust_score'])
                
            if 'accuracy' in m:
                accuracies.append(m['accuracy'])
                
            if 'training_time' in m:
                training_times.append(m['training_time'])
                
        # Calculate statistics
        result = {
            'client_id': client_id,
            'metrics_count': len(metrics),
            'first_seen': min(timestamps) if timestamps else None,
            'last_seen': max(timestamps) if timestamps else None
        }
        
        # Add trust statistics if available
        if trust_scores:
            result['trust_statistics'] = {
                'mean': float(np.mean(trust_scores)),
                'std': float(np.std(trust_scores)),
                'min': float(np.min(trust_scores)),
                'max': float(np.max(trust_scores))
            }
            
        # Add accuracy statistics if available
        if accuracies:
            result['accuracy_statistics'] = {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'declining'
            }
            
        # Add training time statistics if available
        if training_times:
            result['training_time_statistics'] = {
                'mean': float(np.mean(training_times)),
                'std': float(np.std(training_times)),
                'min': float(np.min(training_times)),
                'max': float(np.max(training_times))
            }
            
        return result
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Calculate moving average of data.
        
        Args:
            data: Input data array
            window_size: Window size for moving average
            
        Returns:
            Moving average array
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    def _detect_significant_changes(
        self, 
        data: np.ndarray, 
        window_size: int, 
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect significant changes in time series data.
        
        Args:
            data: Input data array
            window_size: Window size for change detection
            threshold: Change threshold
            
        Returns:
            List of dictionaries with change information
        """
        if len(data) < window_size * 2:
            return []
            
        changes = []
        
        for i in range(window_size, len(data) - window_size):
            before = np.mean(data[i-window_size:i])
            after = np.mean(data[i:i+window_size])
            change = after - before
            
            if abs(change) > threshold:
                changes.append({
                    'index': i,
                    'change': float(change),
                    'before': float(before),
                    'after': float(after),
                    'percent_change': float(change / before * 100 if before != 0 else float('inf'))
                })
                
        return changes
