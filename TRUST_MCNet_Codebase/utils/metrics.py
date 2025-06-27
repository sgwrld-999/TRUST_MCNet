"""
Metrics and evaluation utilities for TRUST-MCNet federated learning.

This module provides comprehensive evaluation metrics for:
- Classification performance (accuracy, precision, recall, F1-score)
- Anomaly detection metrics (AUC, TPR, FPR, detection rate)
- Federated learning specific metrics (trust scores, convergence)
- Model interpretability metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict


class ModelMetrics:
    """
    Comprehensive metrics calculator for federated learning models.
    
    Supports both binary and multiclass classification metrics,
    with special focus on anomaly detection performance.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None,
                                       average: str = 'binary') -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy for multiclass ('binary', 'macro', 'micro', 'weighted')
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(np.unique(y_true)) == 2 else (0, 0, 0, 0)
        
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        
        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['tpr'] = metrics['sensitivity']  # True Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        
        # ROC AUC and Precision-Recall AUC (if probabilities available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                else:
                    # Multiclass AUC
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except ValueError as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    def calculate_anomaly_detection_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate metrics specific to anomaly detection tasks.
        
        Args:
            y_true: True labels (0 for normal, 1 for anomaly)
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional)
            
        Returns:
            Dictionary containing anomaly detection metrics
        """
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_scores)
        
        # Additional anomaly detection specific metrics
        total_anomalies = np.sum(y_true)
        total_normal = len(y_true) - total_anomalies
        
        metrics['detection_rate'] = metrics['recall']  # Same as sensitivity for anomaly detection
        metrics['false_alarm_rate'] = metrics['fpr']
        metrics['anomaly_ratio'] = total_anomalies / len(y_true) if len(y_true) > 0 else 0.0
        
        # Balanced accuracy for imbalanced datasets
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def calculate_loss_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                              loss_fn: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """
        Calculate various loss metrics.
        
        Args:
            y_true: True labels tensor
            y_pred: Predicted logits/probabilities tensor
            loss_fn: Custom loss function (optional)
            
        Returns:
            Dictionary containing loss metrics
        """
        metrics = {}
        
        # Convert to appropriate format
        if y_true.dim() == 1 and y_pred.dim() == 2:
            # Classification case
            if loss_fn is None:
                loss_fn = torch.nn.CrossEntropyLoss()
            
            metrics['cross_entropy_loss'] = loss_fn(y_pred, y_true).item()
            
            # Additional loss calculations
            with torch.no_grad():
                probs = F.softmax(y_pred, dim=1)
                log_probs = F.log_softmax(y_pred, dim=1)
                
                # Negative log likelihood
                metrics['nll_loss'] = F.nll_loss(log_probs, y_true).item()
                
                # Entropy of predictions (uncertainty measure)
                entropy = -torch.sum(probs * log_probs, dim=1)
                metrics['prediction_entropy'] = torch.mean(entropy).item()
                
        elif y_true.dim() == y_pred.dim():
            # Regression case
            metrics['mse_loss'] = F.mse_loss(y_pred, y_true).item()
            metrics['mae_loss'] = F.l1_loss(y_pred, y_true).item()
        
        return metrics
    
    def calculate_federated_metrics(self, client_metrics: Dict[str, Dict[str, float]],
                                   trust_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate federated learning specific metrics.
        
        Args:
            client_metrics: Performance metrics for each client
            trust_scores: Trust scores for each client
            
        Returns:
            Dictionary containing federated learning metrics
        """
        if not client_metrics:
            return {}
        
        metrics = {}
        
        # Aggregate client metrics
        metric_names = set()
        for client_metric in client_metrics.values():
            metric_names.update(client_metric.keys())
        
        for metric_name in metric_names:
            values = [client_metric.get(metric_name, 0.0) for client_metric in client_metrics.values()]
            if values:
                metrics[f'avg_{metric_name}'] = np.mean(values)
                metrics[f'std_{metric_name}'] = np.std(values)
                metrics[f'min_{metric_name}'] = np.min(values)
                metrics[f'max_{metric_name}'] = np.max(values)
        
        # Trust-related metrics
        if trust_scores:
            trust_values = list(trust_scores.values())
            metrics['avg_trust_score'] = np.mean(trust_values)
            metrics['std_trust_score'] = np.std(trust_values)
            metrics['min_trust_score'] = np.min(trust_values)
            metrics['max_trust_score'] = np.max(trust_values)
            
            # Trust distribution analysis
            high_trust_clients = sum(1 for score in trust_values if score > 0.7)
            low_trust_clients = sum(1 for score in trust_values if score < 0.3)
            
            metrics['high_trust_ratio'] = high_trust_clients / len(trust_values)
            metrics['low_trust_ratio'] = low_trust_clients / len(trust_values)
        
        # Client participation metrics
        metrics['num_participating_clients'] = len(client_metrics)
        
        return metrics
    
    def calculate_convergence_metrics(self, loss_history: List[float],
                                    accuracy_history: List[float],
                                    window_size: int = 5) -> Dict[str, float]:
        """
        Calculate convergence-related metrics for federated training.
        
        Args:
            loss_history: History of loss values over rounds
            accuracy_history: History of accuracy values over rounds
            window_size: Window size for trend analysis
            
        Returns:
            Dictionary containing convergence metrics
        """
        metrics = {}
        
        if len(loss_history) < 2:
            return metrics
        
        # Basic statistics
        metrics['final_loss'] = loss_history[-1]
        metrics['final_accuracy'] = accuracy_history[-1] if accuracy_history else 0.0
        metrics['best_loss'] = min(loss_history)
        metrics['best_accuracy'] = max(accuracy_history) if accuracy_history else 0.0
        
        # Convergence rate (loss reduction)
        if len(loss_history) >= window_size:
            recent_losses = loss_history[-window_size:]
            early_losses = loss_history[:window_size]
            
            metrics['loss_reduction'] = np.mean(early_losses) - np.mean(recent_losses)
            metrics['loss_improvement_rate'] = metrics['loss_reduction'] / len(loss_history)
        
        # Stability metrics (variance in recent rounds)
        if len(loss_history) >= window_size:
            recent_loss_variance = np.var(loss_history[-window_size:])
            metrics['loss_stability'] = 1.0 / (1.0 + recent_loss_variance)  # Higher is more stable
        
        # Trend analysis
        if len(loss_history) >= 3:
            # Simple linear trend
            x = np.arange(len(loss_history))
            loss_trend = np.polyfit(x, loss_history, 1)[0]  # Slope
            metrics['loss_trend'] = loss_trend  # Negative means decreasing (good)
            
            if accuracy_history and len(accuracy_history) >= 3:
                acc_trend = np.polyfit(x, accuracy_history, 1)[0]
                metrics['accuracy_trend'] = acc_trend  # Positive means increasing (good)
        
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     target_names: Optional[List[str]] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Formatted classification report string
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           normalize: Optional[str] = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        return cm


class MetricsTracker:
    """
    Tracks and aggregates metrics across federated learning rounds.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.round_metrics = defaultdict(list)
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        self.logger = logging.getLogger(__name__)
    
    def add_round_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Add metrics for a specific round."""
        for key, value in metrics.items():
            self.round_metrics[key].append(value)
        self.logger.debug(f"Added metrics for round {round_num}")
    
    def add_client_metrics(self, client_id: str, round_num: int, metrics: Dict[str, float]):
        """Add metrics for a specific client and round."""
        for key, value in metrics.items():
            self.client_metrics[client_id][key].append(value)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric across rounds."""
        return self.round_metrics.get(metric_name, [])
    
    def get_client_metric_history(self, client_id: str, metric_name: str) -> List[float]:
        """Get history of a specific metric for a specific client."""
        return self.client_metrics.get(client_id, {}).get(metric_name, [])
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}
        
        for metric_name, values in self.round_metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1] if values else 0.0
                }
        
        return summary
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all tracked metrics for analysis."""
        return {
            'round_metrics': dict(self.round_metrics),
            'client_metrics': {k: dict(v) for k, v in self.client_metrics.items()}
        }
