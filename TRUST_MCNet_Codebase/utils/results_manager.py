"""
Results management and visualization for TRUST-MCNet federated learning experiments.

This module provides comprehensive logging, plotting, and analysis capabilities
for federated learning experiments with MNIST dataset.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import warnings

# Optional SHAP import with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class ExperimentLogger:
    """
    Comprehensive experiment logging and metrics tracking.
    """
    
    def __init__(self, experiment_name: str, results_dir: str = "results"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            results_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.metrics = {
            "round_metrics": [],
            "client_weights": [],
            "training_logs": [],
            "final_metrics": {},
            "dataset_stats": {},
            "experiment_config": {}
        }
        
        self.logger.info(f"Initialized experiment logger for {experiment_name}")
        self.logger.info(f"Results will be saved to: {self.experiment_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"experiment_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """Log metrics for a specific round."""
        metrics["round"] = round_num
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics["round_metrics"].append(metrics)
        
        self.logger.info(f"Round {round_num} metrics: {json.dumps(metrics, indent=2)}")
    
    def log_client_weights(self, round_num: int, weights: Dict[str, float]):
        """Log client weights for a specific round."""
        weight_data = {
            "round": round_num,
            "weights": weights,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["client_weights"].append(weight_data)
        
        self.logger.info(f"Round {round_num} client weights: {weights}")
    
    def log_training_event(self, event_type: str, details: Dict[str, Any]):
        """Log a training event."""
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.metrics["training_logs"].append(log_entry)
        
        self.logger.info(f"Training event [{event_type}]: {details}")
    
    def log_dataset_stats(self, stats: Dict[str, Any]):
        """Log dataset statistics."""
        self.metrics["dataset_stats"] = stats
        self.logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.metrics["experiment_config"] = config
        self.logger.info(f"Experiment configuration logged")
    
    def log_final_metrics(self, metrics: Dict[str, Any]):
        """Log final experiment metrics."""
        self.metrics["final_metrics"] = metrics
        self.logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def get_experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir


class ResultsVisualizer:
    """
    Comprehensive visualization for federated learning results.
    """
    
    def __init__(self, experiment_logger: ExperimentLogger):
        """
        Initialize results visualizer.
        
        Args:
            experiment_logger: Experiment logger instance
        """
        self.logger = experiment_logger
        self.experiment_dir = experiment_logger.get_experiment_dir()
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, save_format: str = "png"):
        """Plot training and validation curves over rounds."""
        if not self.logger.metrics["round_metrics"]:
            print("No round metrics available for plotting")
            return
        
        # Extract data
        rounds = []
        train_acc = []
        train_loss = []
        test_acc = []
        test_f1 = []
        test_detection_rate = []
        
        for metric in self.logger.metrics["round_metrics"]:
            rounds.append(metric.get("round", 0))
            train_acc.append(metric.get("train_accuracy", 0))
            train_loss.append(metric.get("train_loss", 0))
            test_acc.append(metric.get("test_accuracy", 0))
            test_f1.append(metric.get("test_f1_score", 0))
            test_detection_rate.append(metric.get("detection_rate", 0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TRUST-MCNet Training Progress', fontsize=16)
        
        # Training accuracy and loss
        axes[0, 0].plot(rounds, train_acc, 'b-', label='Training Accuracy', marker='o')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training Accuracy vs Round')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(rounds, train_loss, 'r-', label='Training Loss', marker='s')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss vs Round')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Test metrics
        axes[1, 0].plot(rounds, test_acc, 'g-', label='Test Accuracy', marker='^')
        axes[1, 0].plot(rounds, test_f1, 'orange', label='F1-Score', marker='d')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Test Metrics vs Round')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Detection rate
        axes[1, 1].plot(rounds, test_detection_rate, 'purple', label='Detection Rate', marker='*')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Detection Rate')
        axes[1, 1].set_title('Anomaly Detection Rate vs Round')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"training_curves.{save_format}"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_file}")
    
    def plot_weight_distribution(self, save_format: str = "png"):
        """Plot client weight distribution analysis."""
        if not self.logger.metrics["client_weights"]:
            print("No client weights available for plotting")
            return
        
        # Extract weight data
        all_weights = []
        rounds = []
        
        for weight_data in self.logger.metrics["client_weights"]:
            round_num = weight_data["round"]
            weights = weight_data["weights"]
            
            for client_id, weight in weights.items():
                all_weights.append({
                    "round": round_num,
                    "client_id": client_id,
                    "weight": weight
                })
                rounds.append(round_num)
        
        if not all_weights:
            print("No weight data available")
            return
        
        df_weights = pd.DataFrame(all_weights)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Client Weight Analysis', fontsize=16)
        
        # Weight histogram
        axes[0, 0].hist(df_weights['weight'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Client Weights')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weight evolution over rounds
        for client_id in df_weights['client_id'].unique():
            client_data = df_weights[df_weights['client_id'] == client_id]
            axes[0, 1].plot(client_data['round'], client_data['weight'], 
                           marker='o', label=f'Client {client_id}', alpha=0.7)
        
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].set_title('Weight Evolution by Client')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot of weights by round
        rounds_to_plot = sorted(df_weights['round'].unique())[-10:]  # Last 10 rounds
        df_recent = df_weights[df_weights['round'].isin(rounds_to_plot)]
        
        if len(df_recent) > 0:
            df_recent.boxplot(column='weight', by='round', ax=axes[1, 0])
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Weight Distribution by Round (Last 10)')
            plt.suptitle('')  # Remove automatic title
        
        # Weight statistics table
        weight_stats = df_weights.groupby('round')['weight'].agg(['mean', 'std', 'min', 'max'])
        
        # Plot weight statistics
        axes[1, 1].plot(weight_stats.index, weight_stats['mean'], 'b-', label='Mean', marker='o')
        axes[1, 1].fill_between(weight_stats.index, 
                               weight_stats['mean'] - weight_stats['std'],
                               weight_stats['mean'] + weight_stats['std'],
                               alpha=0.3, color='blue')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_title('Weight Statistics Over Rounds')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"weight_distribution.{save_format}"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save weight table
        weight_table_file = self.plots_dir / "weight_statistics.csv"
        weight_stats.to_csv(weight_table_file)
        
        print(f"Weight distribution plots saved to {plot_file}")
        print(f"Weight statistics saved to {weight_table_file}")
    
    def create_metrics_summary_table(self) -> pd.DataFrame:
        """Create a comprehensive metrics summary table."""
        if not self.logger.metrics["final_metrics"]:
            print("No final metrics available")
            return pd.DataFrame()
        
        # Extract final metrics
        final_metrics = self.logger.metrics["final_metrics"]
        
        # Create summary table
        summary_data = []
        
        # Federated model metrics
        if "federated" in final_metrics:
            fed_metrics = final_metrics["federated"]
            summary_data.append({
                "Model Type": "Federated (Random Weights)",
                "Accuracy": fed_metrics.get("accuracy", 0),
                "Precision": fed_metrics.get("precision", 0),
                "Recall": fed_metrics.get("recall", 0),
                "F1-Score": fed_metrics.get("f1_score", 0),
                "Detection Rate": fed_metrics.get("detection_rate", 0)
            })
        
        # Centralized model metrics
        if "centralized" in final_metrics:
            cent_metrics = final_metrics["centralized"]
            summary_data.append({
                "Model Type": "Centralized (Baseline)",
                "Accuracy": cent_metrics.get("accuracy", 0),
                "Precision": cent_metrics.get("precision", 0),
                "Recall": cent_metrics.get("recall", 0),
                "F1-Score": cent_metrics.get("f1_score", 0),
                "Detection Rate": cent_metrics.get("detection_rate", 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_file = self.plots_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Metrics summary saved to {summary_file}")
        return summary_df
    
    def create_shap_analysis(self, model: nn.Module, test_data: torch.utils.data.DataLoader, 
                           num_samples: int = 100, save_format: str = "png"):
        """Create SHAP analysis for model explainability."""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping explainability analysis.")
            return
        
        try:
            print("Generating SHAP analysis...")
            
            # Prepare data for SHAP
            model.eval()
            device = next(model.parameters()).device
            
            # Collect samples
            X_samples = []
            y_samples = []
            
            with torch.no_grad():
                for i, (X_batch, y_batch) in enumerate(test_data):
                    if len(X_samples) >= num_samples:
                        break
                    
                    X_batch = X_batch.to(device)
                    X_samples.append(X_batch)
                    y_samples.append(y_batch)
            
            if not X_samples:
                print("No test data available for SHAP analysis")
                return
            
            # Combine samples
            X_combined = torch.cat(X_samples, dim=0)[:num_samples]
            y_combined = torch.cat(y_samples, dim=0)[:num_samples]
            
            # Convert to numpy
            X_np = X_combined.cpu().numpy()
            y_np = y_combined.cpu().numpy()
            
            # Create SHAP explainer
            def model_predict(x):
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(device)
                    outputs = model(x_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
            
            # Use a subset for background
            background = X_np[:min(50, len(X_np))]
            explainer = shap.Explainer(model_predict, background)
            
            # Calculate SHAP values for anomalous predictions
            anomaly_mask = y_np == 1
            if anomaly_mask.sum() == 0:
                print("No anomalous samples found for SHAP analysis")
                return
            
            X_anomalies = X_np[anomaly_mask][:min(20, anomaly_mask.sum())]
            
            shap_values = explainer(X_anomalies)
            
            # Create SHAP plots
            plt.figure(figsize=(12, 8))
            
            # Summary plot
            shap.summary_plot(shap_values[:, :, 1], X_anomalies, 
                            feature_names=[f"Pixel_{i}" for i in range(X_anomalies.shape[1])],
                            show=False, max_display=10)
            
            plt.title('SHAP Summary Plot - Top Features for Anomaly Detection')
            plt.tight_layout()
            
            # Save plot
            shap_file = self.plots_dir / f"shap_summary.{save_format}"
            plt.savefig(shap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance
            feature_importance = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
            top_features = np.argsort(feature_importance)[-10:][::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(top_features)), feature_importance[top_features])
            plt.xlabel('Feature Index (Pixel)')
            plt.ylabel('Mean |SHAP Value|')
            plt.title('Top 10 Most Important Features for Anomaly Detection')
            plt.xticks(range(len(top_features)), [f"Pixel_{i}" for i in top_features], rotation=45)
            plt.tight_layout()
            
            importance_file = self.plots_dir / f"feature_importance.{save_format}"
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP analysis saved to {shap_file} and {importance_file}")
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")


def calculate_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive detection metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Detection rate (recall for anomalies)
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['detection_rate'] = 0
        metrics['false_positive_rate'] = 0
        metrics['true_negative_rate'] = 0
    
    return metrics
