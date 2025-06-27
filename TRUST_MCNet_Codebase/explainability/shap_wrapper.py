"""
SHAP-based explainability wrapper for TRUST-MCNet models.

This module provides interpretability and explainability capabilities
for federated learning models using SHAP (SHapley Additive exPlanations).
Supports feature importance analysis, decision explanation, and trust transparency.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")


class SHAPWrapper:
    """
    SHAP-based explainability wrapper for PyTorch models.
    
    Provides model interpretability through feature importance analysis,
    decision explanations, and visualization capabilities.
    """
    
    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP wrapper.
        
        Args:
            model: PyTorch model to explain
            feature_names: Names of input features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability features. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def initialize_explainer(self, background_data: torch.Tensor, 
                           explainer_type: str = "deep") -> None:
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP baseline
            explainer_type: Type of SHAP explainer ('deep', 'gradient', 'kernel')
        """
        self.background_data = background_data
        
        # Convert to numpy if needed
        if isinstance(background_data, torch.Tensor):
            background_np = background_data.detach().cpu().numpy()
        else:
            background_np = background_data
        
        try:
            if explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, background_data)
            elif explainer_type == "gradient":
                self.explainer = shap.GradientExplainer(self.model, background_data)
            elif explainer_type == "kernel":
                # For kernel explainer, we need a prediction function
                def predict_fn(x):
                    with torch.no_grad():
                        if isinstance(x, np.ndarray):
                            x = torch.FloatTensor(x)
                        return self.model(x).cpu().numpy()
                
                self.explainer = shap.KernelExplainer(predict_fn, background_np)
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
            
            self.logger.info(f"Initialized {explainer_type} SHAP explainer")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain_predictions(self, test_data: torch.Tensor, 
                          max_samples: int = 100) -> shap.Explanation:
        """
        Generate SHAP explanations for test predictions.
        
        Args:
            test_data: Test data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Limit number of samples for computational efficiency
        if len(test_data) > max_samples:
            indices = torch.randperm(len(test_data))[:max_samples]
            test_sample = test_data[indices]
        else:
            test_sample = test_data
        
        try:
            # Generate SHAP values
            shap_values = self.explainer.shap_values(test_sample)
            
            self.logger.info(f"Generated SHAP explanations for {len(test_sample)} samples")
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP explanations: {e}")
            raise
    
    def plot_feature_importance(self, shap_values: Union[np.ndarray, List[np.ndarray]], 
                              test_data: torch.Tensor,
                              plot_type: str = "bar",
                              max_features: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance using SHAP values.
        
        Args:
            shap_values: SHAP values from explain_predictions
            test_data: Corresponding test data
            plot_type: Type of plot ('bar', 'beeswarm', 'waterfall')
            max_features: Maximum number of features to show
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 8))
            
            if plot_type == "bar":
                shap.summary_plot(shap_values, test_data.cpu().numpy(), 
                                plot_type="bar",
                                feature_names=self.feature_names,
                                max_display=max_features,
                                show=False)
            elif plot_type == "beeswarm":
                shap.summary_plot(shap_values, test_data.cpu().numpy(),
                                feature_names=self.feature_names,
                                max_display=max_features,
                                show=False)
            elif plot_type == "waterfall":
                # Waterfall plot for single instance
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first class for multiclass
                
                shap.waterfall_plot(
                    shap.Explanation(values=shap_values[0], 
                                   base_values=self.explainer.expected_value,
                                   data=test_data[0].cpu().numpy(),
                                   feature_names=self.feature_names),
                    show=False
                )
            
            plt.title(f'SHAP Feature Importance - {plot_type.title()} Plot')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved SHAP plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP plot: {e}")
            raise
    
    def explain_single_prediction(self, sample: torch.Tensor, 
                                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Provide detailed explanation for a single prediction.
        
        Args:
            sample: Single sample to explain (1D tensor)
            class_names: Names of output classes
            
        Returns:
            Dictionary containing explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Ensure sample is 2D (batch_size=1)
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        
        try:
            # Get prediction
            with torch.no_grad():
                prediction = self.model(sample)
                predicted_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(sample)
            
            # Handle multiclass case
            if isinstance(shap_values, list):
                # Use SHAP values for predicted class
                relevant_shap_values = shap_values[predicted_class][0]
            else:
                relevant_shap_values = shap_values[0]
            
            # Create feature importance ranking
            feature_importance = []
            for i, (shap_val, feature_val) in enumerate(zip(relevant_shap_values, sample[0])):
                feature_name = self.feature_names[i] if self.feature_names else f"Feature_{i}"
                feature_importance.append({
                    'feature_name': feature_name,
                    'feature_value': feature_val.item(),
                    'shap_value': shap_val,
                    'abs_shap_value': abs(shap_val)
                })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            
            explanation = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'prediction_probabilities': torch.softmax(prediction, dim=1)[0].tolist(),
                'feature_importance': feature_importance,
                'top_positive_features': [f for f in feature_importance if f['shap_value'] > 0][:5],
                'top_negative_features': [f for f in feature_importance if f['shap_value'] < 0][:5],
                'base_value': float(self.explainer.expected_value[predicted_class] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
            }
            
            if class_names:
                explanation['predicted_class_name'] = class_names[predicted_class]
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to explain single prediction: {e}")
            raise
    
    def analyze_feature_interactions(self, test_data: torch.Tensor,
                                   max_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            test_data: Test data for interaction analysis
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary containing interaction analysis results
        """
        if not hasattr(self.explainer, 'shap_interaction_values'):
            self.logger.warning("Interaction values not supported by current explainer")
            return {}
        
        # Limit samples for computational efficiency
        if len(test_data) > max_samples:
            indices = torch.randperm(len(test_data))[:max_samples]
            test_sample = test_data[indices]
        else:
            test_sample = test_data
        
        try:
            interaction_values = self.explainer.shap_interaction_values(test_sample)
            
            # Calculate mean interaction effects
            mean_interactions = np.mean(np.abs(interaction_values), axis=0)
            
            # Find top feature pairs
            n_features = mean_interactions.shape[0]
            top_interactions = []
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction_strength = mean_interactions[i, j]
                    feature_i = self.feature_names[i] if self.feature_names else f"Feature_{i}"
                    feature_j = self.feature_names[j] if self.feature_names else f"Feature_{j}"
                    
                    top_interactions.append({
                        'feature_i': feature_i,
                        'feature_j': feature_j,
                        'interaction_strength': interaction_strength
                    })
            
            # Sort by interaction strength
            top_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            results = {
                'interaction_matrix': mean_interactions,
                'top_interactions': top_interactions[:20],  # Top 20 interactions
                'feature_names': self.feature_names or [f"Feature_{i}" for i in range(n_features)]
            }
            
            self.logger.info(f"Analyzed feature interactions for {len(test_sample)} samples")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feature interactions: {e}")
            return {}
    
    def generate_trust_explanation(self, client_data: torch.Tensor,
                                 trust_score: float,
                                 client_id: str) -> Dict[str, Any]:
        """
        Generate explanation for client trust score based on model behavior.
        
        Args:
            client_data: Client's data sample
            trust_score: Client's trust score
            client_id: Client identifier
            
        Returns:
            Dictionary containing trust explanation
        """
        try:
            # Get SHAP explanations for client data
            shap_values = self.explain_predictions(client_data, max_samples=min(50, len(client_data)))
            
            # Analyze consistency of explanations
            if isinstance(shap_values, list):
                # Multiclass case - use first class for analysis
                values_array = np.array(shap_values[0])
            else:
                values_array = shap_values
            
            # Calculate explanation statistics
            mean_shap = np.mean(np.abs(values_array), axis=0)
            std_shap = np.std(values_array, axis=0)
            
            # Feature consistency (lower variance indicates more consistent explanations)
            consistency_score = 1.0 / (1.0 + np.mean(std_shap))
            
            # Identify most important features for this client
            top_features_idx = np.argsort(mean_shap)[-10:]  # Top 10 features
            top_features = []
            
            for idx in reversed(top_features_idx):
                feature_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
                top_features.append({
                    'feature_name': feature_name,
                    'importance': mean_shap[idx],
                    'consistency': 1.0 / (1.0 + std_shap[idx])
                })
            
            trust_explanation = {
                'client_id': client_id,
                'trust_score': trust_score,
                'explanation_consistency': consistency_score,
                'top_features': top_features,
                'overall_feature_importance': mean_shap.tolist(),
                'feature_consistency': (1.0 / (1.0 + std_shap)).tolist(),
                'trust_factors': {
                    'model_consistency': consistency_score,
                    'feature_reliability': np.mean(1.0 / (1.0 + std_shap)),
                    'prediction_stability': float(1.0 - np.mean(std_shap) / (np.mean(mean_shap) + 1e-6))
                }
            }
            
            return trust_explanation
            
        except Exception as e:
            self.logger.error(f"Failed to generate trust explanation: {e}")
            return {'client_id': client_id, 'trust_score': trust_score, 'error': str(e)}
    
    def create_explanation_dashboard(self, explanations: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive explanation dashboard.
        
        Args:
            explanations: Dictionary containing various explanation results
            save_path: Path to save the dashboard
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('TRUST-MCNet Model Explanations Dashboard', fontsize=16)
            
            # Plot 1: Feature importance
            if 'feature_importance' in explanations:
                importance_data = explanations['feature_importance']
                features = [item['feature_name'] for item in importance_data[:10]]
                values = [item['abs_shap_value'] for item in importance_data[:10]]
                
                axes[0, 0].barh(features, values)
                axes[0, 0].set_title('Top 10 Feature Importance')
                axes[0, 0].set_xlabel('Absolute SHAP Value')
            
            # Plot 2: Trust scores distribution
            if 'trust_scores' in explanations:
                trust_scores = explanations['trust_scores']
                axes[0, 1].hist(trust_scores, bins=20, alpha=0.7)
                axes[0, 1].set_title('Trust Scores Distribution')
                axes[0, 1].set_xlabel('Trust Score')
                axes[0, 1].set_ylabel('Frequency')
            
            # Plot 3: Feature interactions heatmap
            if 'interaction_matrix' in explanations:
                interaction_matrix = explanations['interaction_matrix']
                im = axes[1, 0].imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
                axes[1, 0].set_title('Feature Interactions Heatmap')
                plt.colorbar(im, ax=axes[1, 0])
            
            # Plot 4: Prediction confidence
            if 'prediction_confidence' in explanations:
                confidence_data = explanations['prediction_confidence']
                axes[1, 1].plot(confidence_data)
                axes[1, 1].set_title('Prediction Confidence Over Time')
                axes[1, 1].set_xlabel('Sample Index')
                axes[1, 1].set_ylabel('Confidence')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved explanation dashboard to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create explanation dashboard: {e}")
            raise
    
    def export_explanations(self, explanations: Dict[str, Any], 
                          export_path: str, format: str = "json") -> None:
        """
        Export explanations to file.
        
        Args:
            explanations: Dictionary containing explanation results
            export_path: Path to export file
            format: Export format ('json', 'csv', 'pickle')
        """
        try:
            if format == "json":
                import json
                # Convert numpy arrays to lists for JSON serialization
                serializable_explanations = self._make_json_serializable(explanations)
                with open(export_path, 'w') as f:
                    json.dump(serializable_explanations, f, indent=2)
            
            elif format == "csv":
                # Export feature importance as CSV
                if 'feature_importance' in explanations:
                    df = pd.DataFrame(explanations['feature_importance'])
                    df.to_csv(export_path, index=False)
            
            elif format == "pickle":
                import pickle
                with open(export_path, 'wb') as f:
                    pickle.dump(explanations, f)
            
            self.logger.info(f"Exported explanations to {export_path} in {format} format")
            
        except Exception as e:
            self.logger.error(f"Failed to export explanations: {e}")
            raise
    
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
