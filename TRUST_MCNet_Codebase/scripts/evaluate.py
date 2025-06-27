"""
Evaluation script for TRUST-MCNet federated learning framework.

This script evaluates the final global model and generates comprehensive
analysis including SHAP explanations, performance metrics, and trust analysis.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models.model import MLPModel, LSTMModel
from data.data_loader import DataLoader
from utils.metrics import ModelMetrics
from explainability.shap_wrapper import SHAPWrapper
from logs.logger import FederatedLogger
from trust_module.trust_evaluator import TrustEvaluator


class ModelEvaluator:
    """
    Comprehensive model evaluation system for TRUST-MCNet.
    
    Evaluates model performance, generates explanations, and analyzes
    trust mechanisms effectiveness.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize model evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calculator = ModelMetrics()
        self.logger = FederatedLogger(
            log_dir=self.config.get('log_dir', 'logs'),
            experiment_name=f"evaluation_{self.config.get('experiment_name', 'trust_mcnet')}"
        ).logger
        
        # Initialize data loader
        self.data_loader = DataLoader(self.config)
        
        # Load test data
        self.test_data = self._load_test_data()
        
        self.logger.info("Initialized model evaluator")
    
    def _load_test_data(self):
        """Load test dataset."""
        try:
            test_data = self.data_loader.load_test_data()
            self.logger.info(f"Loaded test data with {len(test_data)} samples")
            return test_data
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            return None
    
    def load_model(self, model_path: str) -> nn.Module:
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded PyTorch model
        """
        try:
            # Determine model type from config
            model_type = self.config.get('model_type', 'mlp')
            input_size = self.config.get('input_size', 41)
            hidden_sizes = self.config.get('hidden_sizes', [64, 32])
            num_classes = self.config.get('num_classes', 2)
            
            if model_type.lower() == 'mlp':
                model = MLPModel(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    num_classes=num_classes,
                    dropout_rate=self.config.get('dropout_rate', 0.2)
                )
            elif model_type.lower() == 'lstm':
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.config.get('lstm_hidden_size', 64),
                    num_layers=self.config.get('lstm_num_layers', 2),
                    num_classes=num_classes,
                    dropout_rate=self.config.get('dropout_rate', 0.2)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'global_model' in checkpoint:
                model.load_state_dict(checkpoint['global_model'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_model_performance(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model to evaluate
            
        Returns:
            Dictionary containing performance metrics
        """
        if self.test_data is None:
            self.logger.error("Test data not available")
            return {}
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_labels in self.test_data:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate comprehensive metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics = self.metrics_calculator.calculate_anomaly_detection_metrics(
                y_true, y_pred, y_proba[:, 1]
            )
        else:  # Multiclass
            metrics = self.metrics_calculator.calculate_classification_metrics(
                y_true, y_pred, y_proba, average='macro'
            )
        
        self.logger.info("Model performance evaluation completed")
        return metrics
    
    def generate_model_explanations(self, model: nn.Module, 
                                  max_samples: int = 200) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model: Trained model to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            # Get feature names if available
            feature_names = self.config.get('feature_names', None)
            
            # Initialize SHAP wrapper
            shap_wrapper = SHAPWrapper(model, feature_names)
            
            # Prepare background and test data
            if self.test_data is None:
                self.logger.error("Test data not available for explanations")
                return {}
            
            # Get sample data for SHAP
            background_samples = []
            test_samples = []
            
            for batch_features, _ in self.test_data:
                background_samples.append(batch_features)
                test_samples.append(batch_features)
                
                if len(background_samples) * batch_features.shape[0] >= max_samples:
                    break
            
            background_data = torch.cat(background_samples, dim=0)[:max_samples//2]
            test_data = torch.cat(test_samples, dim=0)[:max_samples]
            
            # Initialize explainer
            explainer_type = self.config.get('explainer_type', 'deep')
            shap_wrapper.initialize_explainer(background_data, explainer_type)
            
            # Generate explanations
            shap_values = shap_wrapper.explain_predictions(test_data, max_samples)
            
            # Analyze feature importance
            if isinstance(shap_values, list):
                # Multiclass case
                feature_importance = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance ranking
            importance_ranking = []
            for i, importance in enumerate(feature_importance):
                feature_name = feature_names[i] if feature_names else f"Feature_{i}"
                importance_ranking.append({
                    'feature_name': feature_name,
                    'importance': float(importance),
                    'rank': i + 1
                })
            
            # Sort by importance
            importance_ranking.sort(key=lambda x: x['importance'], reverse=True)
            
            # Update ranks
            for i, item in enumerate(importance_ranking):
                item['rank'] = i + 1
            
            explanations = {
                'shap_values': shap_values,
                'feature_importance': importance_ranking,
                'background_data': background_data.cpu().numpy(),
                'test_data': test_data.cpu().numpy(),
                'feature_names': feature_names
            }
            
            self.logger.info("Generated SHAP explanations")
            return explanations
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanations: {e}")
            return {}
    
    def analyze_trust_mechanisms(self, trust_log_file: str) -> Dict[str, Any]:
        """
        Analyze effectiveness of trust mechanisms.
        
        Args:
            trust_log_file: Path to trust scores log file
            
        Returns:
            Dictionary containing trust analysis results
        """
        try:
            # Load trust data
            trust_df = pd.read_csv(trust_log_file)
            
            # Calculate trust statistics
            trust_stats = {}
            
            # Overall trust statistics
            trust_stats['overall'] = {
                'mean_trust': trust_df['trust_score'].mean(),
                'std_trust': trust_df['trust_score'].std(),
                'min_trust': trust_df['trust_score'].min(),
                'max_trust': trust_df['trust_score'].max()
            }
            
            # Per-client trust evolution
            client_stats = {}
            for client_id in trust_df['client_id'].unique():
                client_data = trust_df[trust_df['client_id'] == client_id]
                
                client_stats[client_id] = {
                    'mean_trust': client_data['trust_score'].mean(),
                    'trust_trend': self._calculate_trend(client_data['trust_score'].values),
                    'trust_volatility': client_data['trust_score'].std(),
                    'final_trust': client_data['trust_score'].iloc[-1],
                    'num_rounds': len(client_data)
                }
            
            trust_stats['clients'] = client_stats
            
            # Trust distribution analysis
            high_trust_clients = len([c for c in client_stats.values() if c['mean_trust'] > 0.7])
            low_trust_clients = len([c for c in client_stats.values() if c['mean_trust'] < 0.3])
            
            trust_stats['distribution'] = {
                'high_trust_clients': high_trust_clients,
                'low_trust_clients': low_trust_clients,
                'total_clients': len(client_stats),
                'high_trust_ratio': high_trust_clients / len(client_stats) if client_stats else 0,
                'low_trust_ratio': low_trust_clients / len(client_stats) if client_stats else 0
            }
            
            self.logger.info("Completed trust mechanism analysis")
            return trust_stats
            
        except Exception as e:
            self.logger.error(f"Failed to analyze trust mechanisms: {e}")
            return {}
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend of values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]
    
    def create_evaluation_plots(self, metrics: Dict[str, float], 
                              explanations: Dict[str, Any],
                              trust_analysis: Dict[str, Any],
                              save_dir: str = "evaluation_plots"):
        """
        Create comprehensive evaluation plots.
        
        Args:
            metrics: Performance metrics
            explanations: SHAP explanation results
            trust_analysis: Trust mechanism analysis
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot 1: Performance metrics bar chart
        if metrics:
            plt.figure(figsize=(12, 6))
            
            # Select key metrics for visualization
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            plot_metrics = {k: v for k, v in metrics.items() if k in key_metrics and v is not None}
            
            if plot_metrics:
                plt.bar(plot_metrics.keys(), plot_metrics.values())
                plt.title('Model Performance Metrics')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for i, (k, v) in enumerate(plot_metrics.items()):
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
                
                plt.tight_layout()
                plt.savefig(save_path / 'performance_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 2: Feature importance
        if explanations and 'feature_importance' in explanations:
            importance_data = explanations['feature_importance'][:15]  # Top 15 features
            
            plt.figure(figsize=(12, 8))
            features = [item['feature_name'] for item in importance_data]
            importance = [item['importance'] for item in importance_data]
            
            plt.barh(features, importance)
            plt.title('Top 15 Feature Importance (SHAP)')
            plt.xlabel('SHAP Value (Absolute)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Trust distribution
        if trust_analysis and 'clients' in trust_analysis:
            client_stats = trust_analysis['clients']
            mean_trusts = [stats['mean_trust'] for stats in client_stats.values()]
            
            plt.figure(figsize=(10, 6))
            plt.hist(mean_trusts, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Client Trust Scores')
            plt.xlabel('Mean Trust Score')
            plt.ylabel('Number of Clients')
            plt.axvline(np.mean(mean_trusts), color='red', linestyle='--', 
                       label=f'Overall Mean: {np.mean(mean_trusts):.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path / 'trust_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Evaluation plots saved to {save_dir}")
    
    def generate_evaluation_report(self, metrics: Dict[str, float],
                                 explanations: Dict[str, Any],
                                 trust_analysis: Dict[str, Any],
                                 output_file: str = "evaluation_report.txt") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Performance metrics
            explanations: SHAP explanation results
            trust_analysis: Trust mechanism analysis
            output_file: Output file path
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("TRUST-MCNet Model Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Timestamp: {pd.Timestamp.now()}")
        report_lines.append(f"Configuration: {self.config.get('experiment_name', 'N/A')}")
        
        # Performance metrics section
        if metrics:
            report_lines.append("\n1. MODEL PERFORMANCE METRICS")
            report_lines.append("-" * 30)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"{metric}: {value:.4f}")
                else:
                    report_lines.append(f"{metric}: {value}")
        
        # Feature importance section
        if explanations and 'feature_importance' in explanations:
            importance_data = explanations['feature_importance']
            report_lines.append("\n2. FEATURE IMPORTANCE ANALYSIS")
            report_lines.append("-" * 30)
            report_lines.append("Top 10 Most Important Features:")
            for i, item in enumerate(importance_data[:10]):
                report_lines.append(f"{i+1:2d}. {item['feature_name']}: {item['importance']:.4f}")
        
        # Trust analysis section
        if trust_analysis:
            report_lines.append("\n3. TRUST MECHANISM ANALYSIS")
            report_lines.append("-" * 30)
            
            if 'overall' in trust_analysis:
                overall_stats = trust_analysis['overall']
                report_lines.append("Overall Trust Statistics:")
                report_lines.append(f"  Mean Trust Score: {overall_stats['mean_trust']:.4f}")
                report_lines.append(f"  Trust Std Dev: {overall_stats['std_trust']:.4f}")
                report_lines.append(f"  Min Trust Score: {overall_stats['min_trust']:.4f}")
                report_lines.append(f"  Max Trust Score: {overall_stats['max_trust']:.4f}")
            
            if 'distribution' in trust_analysis:
                dist_stats = trust_analysis['distribution']
                report_lines.append(f"\nTrust Distribution:")
                report_lines.append(f"  High Trust Clients (>0.7): {dist_stats['high_trust_clients']} ({dist_stats['high_trust_ratio']:.2%})")
                report_lines.append(f"  Low Trust Clients (<0.3): {dist_stats['low_trust_clients']} ({dist_stats['low_trust_ratio']:.2%})")
                report_lines.append(f"  Total Clients: {dist_stats['total_clients']}")
        
        # Model architecture section
        report_lines.append("\n4. MODEL CONFIGURATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Model Type: {self.config.get('model_type', 'N/A')}")
        report_lines.append(f"Input Size: {self.config.get('input_size', 'N/A')}")
        report_lines.append(f"Hidden Sizes: {self.config.get('hidden_sizes', 'N/A')}")
        report_lines.append(f"Number of Classes: {self.config.get('num_classes', 'N/A')}")
        report_lines.append(f"Trust Mode: {self.config.get('trust_mode', 'N/A')}")
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Evaluation report saved to {output_file}")
        return report_content
    
    def run_full_evaluation(self, model_path: str, trust_log_file: str, 
                          output_dir: str = "evaluation_results"):
        """
        Run complete evaluation pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            trust_log_file: Path to trust scores log file
            output_dir: Directory to save evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting full model evaluation")
        
        # Load model
        model = self.load_model(model_path)
        
        # Evaluate performance
        metrics = self.evaluate_model_performance(model)
        
        # Generate explanations
        explanations = self.generate_model_explanations(model)
        
        # Analyze trust mechanisms
        trust_analysis = self.analyze_trust_mechanisms(trust_log_file)
        
        # Create plots
        plots_dir = output_path / "plots"
        self.create_evaluation_plots(metrics, explanations, trust_analysis, str(plots_dir))
        
        # Generate report
        report_file = output_path / "evaluation_report.txt"
        self.generate_evaluation_report(metrics, explanations, trust_analysis, str(report_file))
        
        # Save detailed results
        results = {
            'metrics': metrics,
            'trust_analysis': trust_analysis,
            'config': self.config
        }
        
        import json
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Full evaluation completed. Results saved to {output_dir}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate TRUST-MCNet model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--trust_log", type=str, required=True,
                       help="Path to trust scores log file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(args.config)
    evaluator.run_full_evaluation(args.model, args.trust_log, args.output_dir)


if __name__ == "__main__":
    main()
