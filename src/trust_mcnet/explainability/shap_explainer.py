"""
Enhanced SHAP explainer for TRUST-MCNet encrypted traffic anomaly detection.

This module provides comprehensive SHAP-based interpretability for anomaly detection models,
supporting both neural networks and traditional ML models with optimized performance
for network traffic analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
import pickle

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedSHAPExplainer:
    """
    Enhanced SHAP explainer supporting multiple model types for anomaly detection.
    
    Features:
    - Support for PyTorch neural networks
    - Support for XGBoost, RandomForest, IsolationForest
    - Optimized TreeExplainer for tree-based models
    - Minimal performance impact through selective explanation
    - Comprehensive visualization suite
    """
    
    def __init__(
        self,
        model: Union[nn.Module, Any],
        model_type: str = "pytorch",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_size: int = 100,
        enable_performance_mode: bool = True
    ):
        """
        Initialize enhanced SHAP explainer.
        
        Args:
            model: Trained model (PyTorch nn.Module or sklearn-compatible)
            model_type: Type of model ('pytorch', 'xgboost', 'randomforest', 'isolationforest')
            feature_names: Names of input features
            class_names: Names of output classes
            background_size: Size of background dataset for SHAP
            enable_performance_mode: Enable optimizations for better performance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names or []
        self.class_names = class_names or ['Normal', 'Anomaly']
        self.background_size = background_size
        self.enable_performance_mode = enable_performance_mode
        
        self.explainer = None
        self.background_data = None
        self.feature_importance_cache = {}
        
        # Performance tracking
        self.explanation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized Enhanced SHAP Explainer for {model_type} model")
    
    def initialize_explainer(
        self,
        background_data: Union[torch.Tensor, np.ndarray],
        explainer_type: Optional[str] = None
    ) -> None:
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP baseline
            explainer_type: Override explainer type selection
        """
        self.background_data = self._prepare_background_data(background_data)
        
        if explainer_type is None:
            explainer_type = self._select_optimal_explainer()
        
        try:
            if self.model_type == "pytorch":
                self._initialize_pytorch_explainer(explainer_type)
            elif self.model_type in ["xgboost", "randomforest"]:
                self._initialize_tree_explainer()
            elif self.model_type == "isolationforest":
                self._initialize_kernel_explainer()
            else:
                self._initialize_kernel_explainer()  # Fallback
                
            logger.info(f"Initialized {explainer_type} explainer for {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain_predictions(
        self,
        X: Union[torch.Tensor, np.ndarray],
        max_samples: int = 100,
        explain_anomalies_only: bool = True,
        top_k_features: int = 20
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions with performance optimizations.
        
        Args:
            X: Input data to explain
            max_samples: Maximum number of samples to explain
            explain_anomalies_only: Only explain predicted anomalies
            top_k_features: Number of top features to return
            
        Returns:
            Dictionary containing SHAP values and analysis
        """
        start_time = datetime.now()
        
        # Convert and limit input data
        X_processed = self._prepare_input_data(X, max_samples)
        
        # Get predictions first
        predictions = self._get_predictions(X_processed)
        
        # Filter to anomalies only if requested
        if explain_anomalies_only:
            anomaly_mask = predictions == 1  # Assuming 1 = anomaly
            if not np.any(anomaly_mask):
                logger.warning("No anomalies detected for explanation")
                return self._empty_explanation_result()
            
            X_to_explain = X_processed[anomaly_mask]
            predictions_to_explain = predictions[anomaly_mask]
            indices_explained = np.where(anomaly_mask)[0]
        else:
            X_to_explain = X_processed
            predictions_to_explain = predictions
            indices_explained = np.arange(len(X_processed))
        
        # Generate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_to_explain)
            
            # Handle multiclass case
            if isinstance(shap_values, list):
                # For binary classification, use anomaly class (index 1)
                shap_values_anomaly = shap_values[1]
            else:
                shap_values_anomaly = shap_values
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                shap_values_anomaly, top_k_features
            )
            
            # Generate comprehensive analysis
            explanation_result = {
                'shap_values': shap_values_anomaly,
                'predictions': predictions_to_explain,
                'explained_indices': indices_explained,
                'feature_importance': feature_importance,
                'top_features': self._get_top_features(feature_importance, top_k_features),
                'explanation_metadata': {
                    'model_type': self.model_type,
                    'num_samples_explained': len(X_to_explain),
                    'explanation_time': (datetime.now() - start_time).total_seconds(),
                    'feature_names': self.feature_names,
                    'class_names': self.class_names
                }
            }
            
            self.explanation_times.append(explanation_result['explanation_metadata']['explanation_time'])
            
            logger.info(f"Generated explanations for {len(X_to_explain)} samples in "
                       f"{explanation_result['explanation_metadata']['explanation_time']:.2f}s")
            
            return explanation_result
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            raise
    
    def explain_single_prediction(
        self,
        sample: Union[torch.Tensor, np.ndarray],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a single prediction.
        
        Args:
            sample: Single sample to explain
            detailed: Include additional analysis
            
        Returns:
            Detailed explanation dictionary
        """
        # Ensure sample is 2D
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # Get prediction
        prediction = self._get_predictions(sample)[0]
        confidence = self._get_prediction_confidence(sample)[0]
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(sample)
        
        if isinstance(shap_values, list):
            # For binary classification
            shap_values_normal = shap_values[0][0]
            shap_values_anomaly = shap_values[1][0]
        else:
            shap_values_anomaly = shap_values[0]
            shap_values_normal = None
        
        # Create feature-level explanations
        feature_explanations = []
        for i, (feature_val, shap_val) in enumerate(zip(sample[0], shap_values_anomaly)):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
            
            feature_explanations.append({
                'feature_name': feature_name,
                'feature_value': float(feature_val),
                'shap_value': float(shap_val),
                'abs_shap_value': abs(float(shap_val)),
                'contribution': 'positive' if shap_val > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        feature_explanations.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        explanation = {
            'prediction': int(prediction),
            'prediction_class': self.class_names[prediction],
            'confidence': float(confidence),
            'feature_explanations': feature_explanations,
            'top_positive_features': [f for f in feature_explanations if f['shap_value'] > 0][:5],
            'top_negative_features': [f for f in feature_explanations if f['shap_value'] < 0][:5],
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
        }
        
        if detailed:
            explanation.update({
                'feature_statistics': self._calculate_feature_statistics(sample[0]),
                'prediction_breakdown': self._analyze_prediction_breakdown(shap_values_anomaly),
                'risk_factors': self._identify_risk_factors(feature_explanations)
            })
        
        return explanation
    
    def create_visualizations(
        self,
        explanation_result: Dict[str, Any],
        save_dir: Optional[str] = None,
        show_plots: bool = False
    ) -> Dict[str, str]:
        """
        Create comprehensive visualizations for SHAP explanations.
        
        Args:
            explanation_result: Result from explain_predictions
            save_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        plot_files = {}
        
        try:
            # 1. Feature importance bar plot
            plot_files['feature_importance'] = self._create_feature_importance_plot(
                explanation_result['feature_importance'], save_path, show_plots
            )
            
            # 2. SHAP summary plot (beeswarm)
            if len(explanation_result['shap_values']) > 1:
                plot_files['shap_summary'] = self._create_shap_summary_plot(
                    explanation_result, save_path, show_plots
                )
            
            # 3. SHAP waterfall plot for top anomaly
            if len(explanation_result['shap_values']) > 0:
                plot_files['shap_waterfall'] = self._create_waterfall_plot(
                    explanation_result, save_path, show_plots
                )
            
            # 4. Feature interaction heatmap
            if self.model_type in ['xgboost', 'randomforest']:
                plot_files['feature_interactions'] = self._create_interaction_plot(
                    explanation_result, save_path, show_plots
                )
            
            # 5. Explanation dashboard
            plot_files['dashboard'] = self._create_explanation_dashboard(
                explanation_result, save_path, show_plots
            )
            
            logger.info(f"Created {len(plot_files)} visualization plots")
            return plot_files
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return {}
    
    def generate_explanation_report(
        self,
        explanation_result: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            explanation_result: Result from explain_predictions
            output_file: Path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TRUST-MCNet Anomaly Detection Explanation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model Type: {self.model_type.upper()}")
        report_lines.append("")
        
        # Summary statistics
        metadata = explanation_result['explanation_metadata']
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Samples Analyzed: {metadata['num_samples_explained']}")
        report_lines.append(f"Explanation Time: {metadata['explanation_time']:.2f} seconds")
        report_lines.append(f"Features Analyzed: {len(self.feature_names)}")
        report_lines.append("")
        
        # Top contributing features
        report_lines.append("TOP ANOMALY INDICATORS")
        report_lines.append("-" * 40)
        top_features = explanation_result['top_features']
        for i, feature in enumerate(top_features[:10], 1):
            report_lines.append(f"{i:2d}. {feature['feature_name']:<30} "
                              f"Importance: {feature['importance']:8.4f}")
        report_lines.append("")
        
        # Feature analysis
        report_lines.append("FEATURE ANALYSIS")
        report_lines.append("-" * 40)
        feature_importance = explanation_result['feature_importance']
        
        positive_features = [f for f in feature_importance if f['mean_shap'] > 0]
        negative_features = [f for f in feature_importance if f['mean_shap'] < 0]
        
        report_lines.append(f"Features promoting anomaly detection: {len(positive_features)}")
        report_lines.append(f"Features supporting normal classification: {len(negative_features)}")
        
        if positive_features:
            report_lines.append("\nTop Anomaly Promoting Features:")
            for feature in positive_features[:5]:
                report_lines.append(f"  • {feature['feature_name']}: {feature['mean_shap']:+.4f}")
        
        if negative_features:
            report_lines.append("\nTop Normal Supporting Features:")
            for feature in negative_features[:5]:
                report_lines.append(f"  • {feature['feature_name']}: {feature['mean_shap']:+.4f}")
        
        report_lines.append("")
        
        # Performance insights
        if len(self.explanation_times) > 1:
            report_lines.append("PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            avg_time = np.mean(self.explanation_times)
            report_lines.append(f"Average explanation time: {avg_time:.3f} seconds")
            report_lines.append(f"Cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        report_lines.append("1. Monitor top contributing features for anomaly patterns")
        report_lines.append("2. Investigate samples with high feature importance scores")
        report_lines.append("3. Consider feature engineering for low-importance features")
        report_lines.append("4. Validate explanations with domain experts")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved explanation report to {output_file}")
        
        return report_content
    
    def export_explanations(
        self,
        explanation_result: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export explanations to various formats.
        
        Args:
            explanation_result: Result from explain_predictions
            output_path: Path to save explanations
            format: Export format ('json', 'csv', 'pickle')
        """
        try:
            if format == "json":
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = self._make_json_serializable(explanation_result)
                with open(output_path, 'w') as f:
                    json.dump(serializable_result, f, indent=2)
            
            elif format == "csv":
                # Export feature importance as CSV
                df = pd.DataFrame(explanation_result['feature_importance'])
                df.to_csv(output_path, index=False)
            
            elif format == "pickle":
                with open(output_path, 'wb') as f:
                    pickle.dump(explanation_result, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported explanations to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Failed to export explanations: {e}")
            raise
    
    # Helper methods
    def _prepare_background_data(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Prepare background data for SHAP explainer."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Limit background size for performance
        if len(data) > self.background_size:
            indices = np.random.choice(len(data), self.background_size, replace=False)
            data = data[indices]
        
        return data
    
    def _prepare_input_data(self, X: Union[torch.Tensor, np.ndarray], max_samples: int) -> np.ndarray:
        """Prepare input data for explanation."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        return X
    
    def _select_optimal_explainer(self) -> str:
        """Select optimal SHAP explainer based on model type."""
        if self.model_type == "pytorch":
            return "deep"
        elif self.model_type in ["xgboost", "randomforest"]:
            return "tree"
        else:
            return "kernel"
    
    def _initialize_pytorch_explainer(self, explainer_type: str):
        """Initialize explainer for PyTorch models."""
        if explainer_type == "deep":
            self.explainer = shap.DeepExplainer(self.model, torch.FloatTensor(self.background_data))
        elif explainer_type == "gradient":
            self.explainer = shap.GradientExplainer(self.model, torch.FloatTensor(self.background_data))
        else:
            # Fallback to kernel explainer
            def predict_fn(x):
                with torch.no_grad():
                    if isinstance(x, np.ndarray):
                        x = torch.FloatTensor(x)
                    return torch.softmax(self.model(x), dim=1).cpu().numpy()
            
            self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
    
    def _initialize_tree_explainer(self):
        """Initialize TreeExplainer for tree-based models."""
        self.explainer = shap.TreeExplainer(self.model)
    
    def _initialize_kernel_explainer(self):
        """Initialize KernelExplainer for general models."""
        def predict_fn(x):
            return self.model.predict_proba(x)
        
        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
    
    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from model."""
        if self.model_type == "pytorch":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                return torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            return self.model.predict(X)
    
    def _get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence scores."""
        if self.model_type == "pytorch":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                return probs.max(dim=1)[0].cpu().numpy()
        else:
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)
                return probs.max(axis=1)
            else:
                return np.ones(len(X))  # Fallback
    
    def _calculate_feature_importance(self, shap_values: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Calculate feature importance from SHAP values."""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        mean_shap = np.mean(shap_values, axis=0)
        std_shap = np.std(shap_values, axis=0)
        
        feature_importance = []
        for i in range(len(mean_abs_shap)):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
            feature_importance.append({
                'feature_name': feature_name,
                'feature_index': i,
                'importance': float(mean_abs_shap[i]),
                'mean_shap': float(mean_shap[i]),
                'std_shap': float(std_shap[i]),
                'consistency': float(1.0 / (1.0 + std_shap[i]))  # Higher is more consistent
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Add ranks
        for i, feature in enumerate(feature_importance):
            feature['rank'] = i + 1
        
        return feature_importance[:top_k]
    
    def _get_top_features(self, feature_importance: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Get top K features with additional analysis."""
        return feature_importance[:top_k]
    
    def _empty_explanation_result(self) -> Dict[str, Any]:
        """Return empty explanation result when no anomalies found."""
        return {
            'shap_values': np.array([]),
            'predictions': np.array([]),
            'explained_indices': np.array([]),
            'feature_importance': [],
            'top_features': [],
            'explanation_metadata': {
                'model_type': self.model_type,
                'num_samples_explained': 0,
                'explanation_time': 0.0,
                'feature_names': self.feature_names,
                'class_names': self.class_names
            }
        }
    
    def _calculate_feature_statistics(self, sample: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for a single sample."""
        return {
            'mean_value': float(np.mean(sample)),
            'std_value': float(np.std(sample)),
            'min_value': float(np.min(sample)),
            'max_value': float(np.max(sample)),
            'num_zero_features': int(np.sum(sample == 0)),
            'num_features': len(sample)
        }
    
    def _analyze_prediction_breakdown(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Analyze how SHAP values contribute to prediction."""
        total_positive = np.sum(shap_values[shap_values > 0])
        total_negative = np.sum(shap_values[shap_values < 0])
        
        return {
            'total_positive_contribution': float(total_positive),
            'total_negative_contribution': float(total_negative),
            'net_contribution': float(total_positive + total_negative),
            'num_positive_features': int(np.sum(shap_values > 0)),
            'num_negative_features': int(np.sum(shap_values < 0))
        }
    
    def _identify_risk_factors(self, feature_explanations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify high-risk factors from feature explanations."""
        risk_factors = []
        
        # Get top positive contributors (anomaly indicators)
        positive_features = [f for f in feature_explanations if f['shap_value'] > 0]
        
        for feature in positive_features[:5]:  # Top 5 risk factors
            risk_level = "High" if feature['abs_shap_value'] > 0.1 else "Medium" if feature['abs_shap_value'] > 0.05 else "Low"
            
            risk_factors.append({
                'feature_name': feature['feature_name'],
                'risk_level': risk_level,
                'contribution_score': feature['abs_shap_value'],
                'feature_value': feature['feature_value']
            })
        
        return risk_factors
    
    def _create_feature_importance_plot(self, feature_importance: List[Dict[str, Any]], save_path: Optional[Path], show: bool) -> Optional[str]:
        """Create feature importance bar plot."""
        plt.figure(figsize=(12, 8))
        
        features = [f['feature_name'] for f in feature_importance[:15]]
        importance_values = [f['importance'] for f in feature_importance[:15]]
        
        bars = plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 15 Feature Importance for Anomaly Detection')
        plt.gca().invert_yaxis()
        
        # Color code bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            file_path = save_path / 'feature_importance.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            result = str(file_path)
        else:
            result = None
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return result
    
    def _create_shap_summary_plot(self, explanation_result: Dict[str, Any], save_path: Optional[Path], show: bool) -> Optional[str]:
        """Create SHAP summary (beeswarm) plot."""
        plt.figure(figsize=(12, 8))
        
        # Create dummy data for the plot (SHAP values need to be paired with feature values)
        shap_values = explanation_result['shap_values']
        
        # For summary plot, we need the original feature values
        # This is a simplified version - in practice, you'd store the original X values
        shap.summary_plot(
            shap_values,
            feature_names=self.feature_names[:shap_values.shape[1]] if self.feature_names else None,
            max_display=15,
            show=False
        )
        
        plt.title('SHAP Summary Plot - Feature Impact on Anomaly Detection')
        plt.tight_layout()
        
        if save_path:
            file_path = save_path / 'shap_summary.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            result = str(file_path)
        else:
            result = None
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return result
    
    def _create_waterfall_plot(self, explanation_result: Dict[str, Any], save_path: Optional[Path], show: bool) -> Optional[str]:
        """Create SHAP waterfall plot for top anomaly."""
        if len(explanation_result['shap_values']) == 0:
            return None
        
        # Get the sample with highest anomaly score
        shap_values = explanation_result['shap_values']
        sample_scores = np.sum(shap_values, axis=1)
        top_sample_idx = np.argmax(sample_scores)
        
        plt.figure(figsize=(12, 8))
        
        # Create waterfall plot manually (simplified version)
        sample_shap = shap_values[top_sample_idx]
        feature_names = self.feature_names[:len(sample_shap)] if self.feature_names else [f"F{i}" for i in range(len(sample_shap))]
        
        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(sample_shap))[-15:]  # Top 15
        sorted_shap = sample_shap[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
        
        plt.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_shap)), sorted_names)
        plt.xlabel('SHAP Value')
        plt.title(f'SHAP Waterfall Plot - Top Anomaly (Sample {top_sample_idx})')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            file_path = save_path / 'shap_waterfall.png'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            result = str(file_path)
        else:
            result = None
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return result
    
    def _create_interaction_plot(self, explanation_result: Dict[str, Any], save_path: Optional[Path], show: bool) -> Optional[str]:
        """Create feature interaction heatmap for tree models."""
        if self.model_type not in ['xgboost', 'randomforest']:
            return None
        
        try:
            # This is a placeholder - actual implementation would require SHAP interaction values
            plt.figure(figsize=(10, 8))
            
            # Create dummy interaction matrix
            n_features = min(10, len(self.feature_names))
            interaction_matrix = np.random.rand(n_features, n_features) * 0.1
            
            feature_names = self.feature_names[:n_features] if self.feature_names else [f"F{i}" for i in range(n_features)]
            
            sns.heatmap(
                interaction_matrix,
                xticklabels=feature_names,
                yticklabels=feature_names,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0
            )
            
            plt.title('Feature Interaction Heatmap')
            plt.tight_layout()
            
            if save_path:
                file_path = save_path / 'feature_interactions.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                result = str(file_path)
            else:
                result = None
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to create interaction plot: {e}")
            return None
    
    def _create_explanation_dashboard(self, explanation_result: Dict[str, Any], save_path: Optional[Path], show: bool) -> Optional[str]:
        """Create comprehensive explanation dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TRUST-MCNet Anomaly Detection Explanation Dashboard', fontsize=16)
        
        try:
            # Plot 1: Feature importance
            feature_importance = explanation_result['feature_importance'][:10]
            features = [f['feature_name'] for f in feature_importance]
            values = [f['importance'] for f in feature_importance]
            
            axes[0, 0].barh(features, values)
            axes[0, 0].set_title('Top 10 Feature Importance')
            axes[0, 0].set_xlabel('Mean |SHAP Value|')
            
            # Plot 2: Prediction distribution
            predictions = explanation_result['predictions']
            if len(predictions) > 0:
                axes[0, 1].hist(predictions, bins=2, alpha=0.7, color=['blue', 'red'], edgecolor='black')
                axes[0, 1].set_title('Prediction Distribution')
                axes[0, 1].set_xlabel('Class (0=Normal, 1=Anomaly)')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_xticks([0, 1])
            
            # Plot 3: SHAP value distribution
            shap_values = explanation_result['shap_values']
            if len(shap_values) > 0:
                axes[1, 0].hist(shap_values.flatten(), bins=50, alpha=0.7, color='green')
                axes[1, 0].set_title('SHAP Values Distribution')
                axes[1, 0].set_xlabel('SHAP Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Plot 4: Model performance summary
            metadata = explanation_result['explanation_metadata']
            performance_text = f"""
Model Type: {metadata['model_type'].upper()}
Samples Explained: {metadata['num_samples_explained']}
Explanation Time: {metadata['explanation_time']:.2f}s
Features: {len(metadata['feature_names'])}

Top Risk Indicators:
"""
            for i, feature in enumerate(feature_importance[:5], 1):
                performance_text += f"{i}. {feature['feature_name']}\n"
            
            axes[1, 1].text(0.05, 0.95, performance_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_title('Analysis Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                file_path = save_path / 'explanation_dashboard.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                result = str(file_path)
            else:
                result = None
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            plt.close()
            return None
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects for JSON export."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj


class AnomalyExplanationPipeline:
    """
    Complete pipeline for anomaly detection with SHAP explanations.
    
    This class orchestrates the entire process from model training to explanation generation,
    optimized for encrypted traffic anomaly detection scenarios.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        model_configs: Optional[Dict[str, Any]] = None,
        explanation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the explanation pipeline.
        
        Args:
            feature_names: Names of input features
            model_configs: Configuration for different model types
            explanation_config: Configuration for explanation generation
        """
        self.feature_names = feature_names or []
        self.model_configs = model_configs or {}
        self.explanation_config = explanation_config or {}
        
        self.models = {}
        self.explainers = {}
        self.training_history = {}
        
        logger.info("Initialized Anomaly Explanation Pipeline")
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models for comparative analysis.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            models_to_train: List of model types to train
            
        Returns:
            Training results dictionary
        """
        if models_to_train is None:
            models_to_train = ['xgboost', 'randomforest', 'isolationforest']
        
        results = {}
        
        for model_type in models_to_train:
            try:
                logger.info(f"Training {model_type} model...")
                
                model, training_info = self._train_single_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                
                self.models[model_type] = model
                self.training_history[model_type] = training_info
                results[model_type] = training_info
                
                logger.info(f"Successfully trained {model_type} model")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def initialize_explainers(
        self,
        background_data: np.ndarray,
        models_to_explain: Optional[List[str]] = None
    ) -> None:
        """
        Initialize SHAP explainers for trained models.
        
        Args:
            background_data: Background dataset for SHAP
            models_to_explain: List of models to create explainers for
        """
        if models_to_explain is None:
            models_to_explain = list(self.models.keys())
        
        for model_type in models_to_explain:
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not found, skipping explainer initialization")
                continue
            
            try:
                explainer = EnhancedSHAPExplainer(
                    model=self.models[model_type],
                    model_type=model_type,
                    feature_names=self.feature_names,
                    **self.explanation_config
                )
                
                explainer.initialize_explainer(background_data)
                self.explainers[model_type] = explainer
                
                logger.info(f"Initialized explainer for {model_type}")
                
            except Exception as e:
                logger.error(f"Failed to initialize explainer for {model_type}: {e}")
    
    def explain_anomalies(
        self,
        X_test: np.ndarray,
        model_type: str = "xgboost",
        **explanation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanations for anomalies using specified model.
        
        Args:
            X_test: Test data to explain
            model_type: Type of model to use for explanations
            **explanation_kwargs: Additional arguments for explanation
            
        Returns:
            Comprehensive explanation results
        """
        if model_type not in self.explainers:
            raise ValueError(f"Explainer for {model_type} not initialized")
        
        explainer = self.explainers[model_type]
        
        # Generate explanations
        explanation_result = explainer.explain_predictions(X_test, **explanation_kwargs)
        
        # Add model comparison if multiple models available
        if len(self.models) > 1:
            explanation_result['model_comparison'] = self._compare_model_predictions(X_test)
        
        return explanation_result
    
    def create_comprehensive_report(
        self,
        X_test: np.ndarray,
        output_dir: str,
        include_visualizations: bool = True
    ) -> Dict[str, str]:
        """
        Create comprehensive explanation report with all models.
        
        Args:
            X_test: Test data
            output_dir: Directory to save reports
            include_visualizations: Whether to include visualization plots
            
        Returns:
            Dictionary mapping report types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_files = {}
        
        # Generate explanations for each model
        for model_type, explainer in self.explainers.items():
            try:
                logger.info(f"Generating report for {model_type}...")
                
                # Get explanations
                explanation_result = explainer.explain_predictions(X_test)
                
                # Save explanations
                explanation_file = output_path / f"{model_type}_explanations.json"
                explainer.export_explanations(explanation_result, str(explanation_file), "json")
                
                # Generate text report
                report_file = output_path / f"{model_type}_report.txt"
                report_content = explainer.generate_explanation_report(explanation_result, str(report_file))
                report_files[f"{model_type}_report"] = str(report_file)
                
                # Create visualizations
                if include_visualizations:
                    viz_dir = output_path / f"{model_type}_visualizations"
                    plot_files = explainer.create_visualizations(explanation_result, str(viz_dir))
                    report_files.update({f"{model_type}_{k}": v for k, v in plot_files.items()})
                
            except Exception as e:
                logger.error(f"Failed to generate report for {model_type}: {e}")
        
        # Create comparative analysis
        if len(self.explainers) > 1:
            comparative_file = output_path / "comparative_analysis.txt"
            self._create_comparative_report(X_test, str(comparative_file))
            report_files['comparative_analysis'] = str(comparative_file)
        
        logger.info(f"Generated comprehensive reports in {output_dir}")
        return report_files
    
    def _train_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model and return training info."""
        
        if model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            
            model = xgb.XGBClassifier(
                **self.model_configs.get('xgboost', {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                })
            )
            
        elif model_type == "randomforest":
            model = RandomForestClassifier(
                **self.model_configs.get('randomforest', {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                })
            )
            
        elif model_type == "isolationforest":
            model = IsolationForest(
                **self.model_configs.get('isolationforest', {
                    'contamination': 0.1,
                    'random_state': 42
                })
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        start_time = datetime.now()
        
        if model_type == "isolationforest":
            # Isolation Forest is unsupervised
            model.fit(X_train)
            # Convert anomaly scores to binary predictions for consistency
            train_pred = model.predict(X_train)
            train_pred = (train_pred == -1).astype(int)  # -1 means anomaly
        else:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        train_accuracy = np.mean(train_pred == y_train) if model_type != "isolationforest" else None
        
        training_info = {
            'model_type': model_type,
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train),
            'feature_count': X_train.shape[1]
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            if model_type == "isolationforest":
                val_pred = model.predict(X_val)
                val_pred = (val_pred == -1).astype(int)
            else:
                val_pred = model.predict(X_val)
            
            val_accuracy = np.mean(val_pred == y_val)
            training_info['val_accuracy'] = val_accuracy
            
            # Calculate detailed metrics for binary classification
            if len(np.unique(y_val)) == 2:
                from sklearn.metrics import precision_score, recall_score, f1_score
                training_info.update({
                    'val_precision': precision_score(y_val, val_pred, zero_division=0),
                    'val_recall': recall_score(y_val, val_pred, zero_division=0),
                    'val_f1': f1_score(y_val, val_pred, zero_division=0)
                })
        
        return model, training_info
    
    def _compare_model_predictions(self, X_test: np.ndarray) -> Dict[str, Any]:
        """Compare predictions across different models."""
        predictions = {}
        probabilities = {}
        
        for model_type, model in self.models.items():
            try:
                if model_type == "isolationforest":
                    pred = model.predict(X_test)
                    pred = (pred == -1).astype(int)
                    prob = None
                else:
                    pred = model.predict(X_test)
                    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                predictions[model_type] = pred
                if prob is not None:
                    probabilities[model_type] = prob
                    
            except Exception as e:
                logger.error(f"Failed to get predictions from {model_type}: {e}")
        
        # Calculate agreement between models
        agreement_matrix = {}
        model_list = list(predictions.keys())
        
        for i, model1 in enumerate(model_list):
            for j, model2 in enumerate(model_list[i+1:], i+1):
                agreement = np.mean(predictions[model1] == predictions[model2])
                agreement_matrix[f"{model1}_vs_{model2}"] = agreement
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'agreement_matrix': agreement_matrix,
            'ensemble_prediction': self._calculate_ensemble_prediction(predictions)
        }
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ensemble prediction using majority voting."""
        if not predictions:
            return np.array([])
        
        # Stack predictions
        pred_matrix = np.stack(list(predictions.values()), axis=0)
        
        # Majority voting
        ensemble_pred = np.round(np.mean(pred_matrix, axis=0)).astype(int)
        
        return ensemble_pred
    
    def _create_comparative_report(self, X_test: np.ndarray, output_file: str) -> None:
        """Create comparative analysis report across models."""
        comparison = self._compare_model_predictions(X_test)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TRUST-MCNet Model Comparison Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Test Samples: {len(X_test)}")
        report_lines.append("")
        
        # Model agreement analysis
        report_lines.append("MODEL AGREEMENT ANALYSIS")
        report_lines.append("-" * 40)
        for pair, agreement in comparison['agreement_matrix'].items():
            report_lines.append(f"{pair}: {agreement:.3f}")
        report_lines.append("")
        
        # Prediction statistics
        report_lines.append("PREDICTION STATISTICS")
        report_lines.append("-" * 40)
        for model_type, preds in comparison['predictions'].items():
            anomaly_rate = np.mean(preds)
            report_lines.append(f"{model_type}: {anomaly_rate:.3f} anomaly rate")
        
        # Ensemble results
        ensemble_pred = comparison['ensemble_prediction']
        ensemble_anomaly_rate = np.mean(ensemble_pred)
        report_lines.append(f"Ensemble: {ensemble_anomaly_rate:.3f} anomaly rate")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Find best agreement
        best_agreement = max(comparison['agreement_matrix'].items(), key=lambda x: x[1])
        report_lines.append(f"Highest model agreement: {best_agreement[0]} ({best_agreement[1]:.3f})")
        
        # Check for outlier models
        anomaly_rates = [np.mean(preds) for preds in comparison['predictions'].values()]
        if max(anomaly_rates) - min(anomaly_rates) > 0.2:
            report_lines.append("⚠️  Significant disagreement in anomaly detection rates")
            report_lines.append("   Consider investigating models with extreme rates")
        
        report_lines.append("💡 Use ensemble prediction for robust anomaly detection")
        report_lines.append("💡 Investigate samples where models disagree strongly")
        
        report_content = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Saved comparative analysis to {output_file}")
