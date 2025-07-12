"""
Visualization manager for SHAP explanations and trust metrics.

This module provides comprehensive visualization capabilities for SHAP-based
explanations, trust attribution, and federated learning insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .trust_attribution import ClientExplanation, TrustMetrics, TrustAttributionEngine

logger = logging.getLogger(__name__)


class SHAPVisualizationManager:
    """
    Manager for creating comprehensive SHAP-based visualizations.
    
    Provides both static (matplotlib/seaborn) and interactive (plotly) visualizations
    for SHAP explanations, trust metrics, and federated learning insights.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "visualizations",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = "whitegrid"
    ):
        """
        Initialize visualization manager.
        
        Args:
            feature_names: Names of input features
            output_dir: Directory to save visualizations
            figsize: Default figure size for matplotlib plots
            dpi: DPI for saved figures
            style: Seaborn style for plots
        """
        self.feature_names = feature_names or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up plotting style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            if sns and style:
                sns.set_style(style)
                sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'neutral': '#7f7f7f'
        }
        
    def create_shap_summary_plot(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        title: str = "SHAP Summary Plot",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create SHAP summary plot showing feature importance and effects.
        
        Args:
            shap_values: SHAP explanation values
            X: Original feature values
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available for summary plot")
            return None
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        try:
            plt.figure(figsize=self.figsize)
            
            # Create SHAP summary plot
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return None
    
    def create_feature_importance_plot(
        self,
        feature_importance: np.ndarray,
        title: str = "Feature Importance",
        top_k: int = 15,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create feature importance bar plot.
        
        Args:
            feature_importance: Feature importance scores
            title: Plot title
            top_k: Number of top features to show
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        try:
            # Get top K features
            top_indices = np.argsort(feature_importance)[-top_k:]
            top_scores = feature_importance[top_indices]
            
            # Get feature names
            if self.feature_names:
                top_names = [self.feature_names[i] if i < len(self.feature_names) 
                           else f"Feature_{i}" for i in top_indices]
            else:
                top_names = [f"Feature_{i}" for i in top_indices]
            
            # Create plot
            plt.figure(figsize=self.figsize)
            bars = plt.barh(range(len(top_scores)), top_scores, color=self.colors['primary'])
            
            # Customize plot
            plt.yticks(range(len(top_scores)), top_names)
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, top_scores)):
                plt.text(score + 0.01 * max(top_scores), i, f'{score:.3f}', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return None
    
    def create_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        sample_idx: int = 0,
        title: str = "SHAP Waterfall Plot",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            shap_values: SHAP explanation values
            base_value: Base prediction value
            sample_idx: Index of sample to explain
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available for waterfall plot")
            return None
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        try:
            plt.figure(figsize=self.figsize)
            
            # Create waterfall plot
            if hasattr(shap, 'waterfall_plot'):
                # SHAP v0.40+ style
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=base_value,
                        feature_names=self.feature_names
                    ),
                    show=False
                )
            else:
                # Legacy SHAP style
                shap.waterfall_plot(
                    base_value,
                    shap_values[sample_idx],
                    feature_names=self.feature_names,
                    show=False
                )
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            return None
    
    def create_trust_metrics_dashboard(
        self,
        trust_metrics: Dict[str, TrustMetrics],
        title: str = "Client Trust Metrics Dashboard",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create comprehensive dashboard for trust metrics across clients.
        
        Args:
            trust_metrics: Dictionary of client trust metrics
            title: Dashboard title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        try:
            # Prepare data
            clients = list(trust_metrics.keys())
            metrics_data = {
                'consistency': [trust_metrics[c].explanation_consistency for c in clients],
                'reliability': [trust_metrics[c].prediction_reliability for c in clients],
                'stability': [trust_metrics[c].feature_stability for c in clients],
                'quality': [trust_metrics[c].anomaly_detection_quality for c in clients],
                'overall': [trust_metrics[c].overall_trust_score for c in clients]
            }
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(title, fontsize=20, fontweight='bold')
            
            # Overall trust scores
            ax = axes[0, 0]
            bars = ax.bar(clients, metrics_data['overall'], color=self.colors['primary'])
            ax.set_title('Overall Trust Scores', fontsize=14, fontweight='bold')
            ax.set_ylabel('Trust Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add horizontal line for trust threshold
            ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Trust Threshold')
            ax.legend()
            
            # Individual metrics heatmap
            ax = axes[0, 1]
            metrics_array = np.array([
                metrics_data['consistency'],
                metrics_data['reliability'],
                metrics_data['stability'],
                metrics_data['quality']
            ])
            
            im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(len(clients)))
            ax.set_xticklabels(clients, rotation=45)
            ax.set_yticks(range(4))
            ax.set_yticklabels(['Consistency', 'Reliability', 'Stability', 'Quality'])
            ax.set_title('Trust Metrics Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score')
            
            # Risk assessment pie chart
            ax = axes[0, 2]
            risk_counts = {}
            for metrics in trust_metrics.values():
                risk = metrics.risk_assessment
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            colors_pie = [self.colors['success'], self.colors['warning'], self.colors['danger']]
            ax.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%',
                  colors=colors_pie[:len(risk_counts)])
            ax.set_title('Risk Distribution', fontsize=14, fontweight='bold')
            
            # Consistency vs Reliability scatter
            ax = axes[1, 0]
            ax.scatter(metrics_data['consistency'], metrics_data['reliability'], 
                      c=metrics_data['overall'], cmap='RdYlGn', s=100, alpha=0.7)
            ax.set_xlabel('Explanation Consistency')
            ax.set_ylabel('Prediction Reliability')
            ax.set_title('Consistency vs Reliability', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Stability vs Quality scatter
            ax = axes[1, 1]
            ax.scatter(metrics_data['stability'], metrics_data['quality'], 
                      c=metrics_data['overall'], cmap='RdYlGn', s=100, alpha=0.7)
            ax.set_xlabel('Feature Stability')
            ax.set_ylabel('Anomaly Detection Quality')
            ax.set_title('Stability vs Quality', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Client ranking
            ax = axes[1, 2]
            sorted_clients = sorted(zip(clients, metrics_data['overall']), 
                                  key=lambda x: x[1], reverse=True)
            client_names, scores = zip(*sorted_clients)
            
            colors_rank = [self.colors['success'] if s >= 0.6 else 
                          self.colors['warning'] if s >= 0.4 else 
                          self.colors['danger'] for s in scores]
            
            bars = ax.barh(range(len(client_names)), scores, color=colors_rank)
            ax.set_yticks(range(len(client_names)))
            ax.set_yticklabels(client_names)
            ax.set_xlabel('Trust Score')
            ax.set_title('Client Ranking', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Trust metrics dashboard saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating trust metrics dashboard: {e}")
            return None
    
    def create_trust_trends_plot(
        self,
        trends_data: Dict[str, List],
        client_id: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create time series plot of trust metrics trends.
        
        Args:
            trends_data: Trends data from TrustAttributionEngine
            client_id: Client identifier for the title
            title: Custom plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        if not trends_data:
            logger.warning("No trends data provided")
            return None
        
        try:
            # Convert timestamps if they're strings
            timestamps = trends_data['timestamps']
            if isinstance(timestamps[0], str):
                from datetime import datetime
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot trust score trend
            ax1.plot(timestamps, trends_data['trust_scores'], 
                    color=self.colors['primary'], linewidth=2, marker='o', label='Trust Score')
            ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Trust Threshold')
            ax1.set_ylabel('Trust Score')
            ax1.set_title(title or f'Trust Trends for Client {client_id}', 
                         fontsize=16, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot individual metrics
            ax2.plot(timestamps, trends_data['consistency_scores'], 
                    color=self.colors['success'], linewidth=2, marker='s', 
                    label='Consistency', alpha=0.8)
            ax2.plot(timestamps, trends_data['reliability_scores'], 
                    color=self.colors['warning'], linewidth=2, marker='^', 
                    label='Reliability', alpha=0.8)
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Metric Score')
            ax2.set_title('Individual Metrics Trends', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Format x-axis
            if len(timestamps) > 10:
                ax1.xaxis.set_major_locator(mdates.MaxNLocator(nbins=10))
                ax2.xaxis.set_major_locator(mdates.MaxNLocator(nbins=10))
            
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Trust trends plot saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating trust trends plot: {e}")
            return None
    
    def create_federated_overview_plot(
        self,
        client_explanations: Dict[str, List[ClientExplanation]],
        title: str = "Federated Learning Overview",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Optional[str]:
        """
        Create overview plot for federated learning with client contributions.
        
        Args:
            client_explanations: Dictionary of client explanations
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return None
        
        try:
            # Prepare data
            clients = list(client_explanations.keys())
            client_data = {}
            
            for client_id, explanations in client_explanations.items():
                if explanations:
                    latest_exp = explanations[-1]
                    client_data[client_id] = {
                        'num_explanations': len(explanations),
                        'avg_confidence': np.mean([exp.confidence_score for exp in explanations 
                                                 if exp.confidence_score > 0]),
                        'avg_data_quality': np.mean([exp.data_quality_score for exp in explanations 
                                                   if exp.data_quality_score > 0]),
                        'latest_anomaly_rate': np.mean(latest_exp.predictions),
                        'feature_diversity': np.std(latest_exp.feature_importance)
                    }
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(title, fontsize=20, fontweight='bold')
            
            # Client contributions (number of explanations)
            ax = axes[0, 0]
            contributions = [client_data[c]['num_explanations'] for c in clients]
            bars = ax.bar(clients, contributions, color=self.colors['primary'])
            ax.set_title('Client Contributions', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Explanations')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, contributions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(value)}', ha='center', va='bottom')
            
            # Data quality distribution
            ax = axes[0, 1]
            quality_scores = [client_data[c]['avg_data_quality'] for c in clients 
                            if not np.isnan(client_data[c]['avg_data_quality'])]
            
            if quality_scores:
                ax.hist(quality_scores, bins=10, color=self.colors['success'], alpha=0.7)
                ax.set_title('Data Quality Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Average Data Quality Score')
                ax.set_ylabel('Number of Clients')
            else:
                ax.text(0.5, 0.5, 'No data quality scores available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Data Quality Distribution', fontsize=14, fontweight='bold')
            
            # Anomaly detection rates
            ax = axes[1, 0]
            anomaly_rates = [client_data[c]['latest_anomaly_rate'] for c in clients]
            ax.scatter(range(len(clients)), anomaly_rates, 
                      c=anomaly_rates, cmap='RdYlBu_r', s=100, alpha=0.7)
            ax.set_xticks(range(len(clients)))
            ax.set_xticklabels(clients, rotation=45)
            ax.set_ylabel('Latest Anomaly Rate')
            ax.set_title('Client Anomaly Detection Rates', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Feature diversity
            ax = axes[1, 1]
            diversity_scores = [client_data[c]['feature_diversity'] for c in clients]
            bars = ax.bar(clients, diversity_scores, color=self.colors['secondary'])
            ax.set_title('Feature Importance Diversity', fontsize=14, fontweight='bold')
            ax.set_ylabel('Standard Deviation of Feature Importance')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                full_path = self.output_dir / save_path
                plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Federated overview plot saved to {full_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return str(full_path) if save_path else None
            
        except Exception as e:
            logger.error(f"Error creating federated overview plot: {e}")
            return None
    
    def create_interactive_shap_dashboard(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        predictions: np.ndarray,
        client_id: str = "unknown",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create interactive SHAP dashboard using Plotly.
        
        Args:
            shap_values: SHAP explanation values
            X: Original feature values
            predictions: Model predictions
            client_id: Client identifier
            save_path: Path to save HTML file
            
        Returns:
            Path to saved HTML file if save_path provided
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive plots")
            return None
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Feature Importance',
                    'SHAP Values Distribution',
                    'Prediction vs SHAP Sum',
                    'Feature Interactions'
                ),
                specs=[[{"type": "bar"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # Feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(feature_importance)[-15:]
            
            feature_names_plot = [self.feature_names[i] if i < len(self.feature_names) 
                                else f"Feature_{i}" for i in top_indices]
            
            fig.add_trace(
                go.Bar(
                    x=feature_importance[top_indices],
                    y=feature_names_plot,
                    orientation='h',
                    name='Feature Importance',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # SHAP values distribution
            for i, idx in enumerate(top_indices[-5:]):  # Top 5 features
                feature_name = (self.feature_names[idx] if idx < len(self.feature_names) 
                              else f"Feature_{idx}")
                fig.add_trace(
                    go.Box(
                        y=shap_values[:, idx],
                        name=feature_name,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Prediction vs SHAP sum
            shap_sum = np.sum(shap_values, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=shap_sum,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=predictions,
                        colorscale='RdYlBu',
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Feature correlation heatmap
            correlation_matrix = np.corrcoef(shap_values[:, top_indices[-10:]].T)
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=feature_names_plot[-10:],
                    y=feature_names_plot[-10:],
                    colorscale='RdBu',
                    showscale=True
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text=f"Interactive SHAP Dashboard - Client {client_id}",
                title_x=0.5,
                showlegend=False
            )
            
            # Save HTML file if path provided
            if save_path:
                full_path = self.output_dir / save_path
                fig.write_html(str(full_path))
                logger.info(f"Interactive SHAP dashboard saved to {full_path}")
                return str(full_path)
            
            # Show plot
            fig.show()
            return None
            
        except Exception as e:
            logger.error(f"Error creating interactive SHAP dashboard: {e}")
            return None
    
    def generate_visualization_report(
        self,
        client_explanations: Dict[str, List[ClientExplanation]],
        trust_engine: TrustAttributionEngine,
        output_prefix: str = "trust_mcnet_report"
    ) -> Dict[str, str]:
        """
        Generate comprehensive visualization report with all plots.
        
        Args:
            client_explanations: Dictionary of client explanations
            trust_engine: Trust attribution engine
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        generated_plots = {}
        
        try:
            # Compute trust metrics for all clients
            trust_metrics = trust_engine.compute_trust_metrics()
            
            # Trust metrics dashboard
            dashboard_path = f"{output_prefix}_trust_dashboard.png"
            result = self.create_trust_metrics_dashboard(
                trust_metrics, save_path=dashboard_path, show_plot=False
            )
            if result:
                generated_plots['trust_dashboard'] = result
            
            # Federated overview
            overview_path = f"{output_prefix}_federated_overview.png"
            result = self.create_federated_overview_plot(
                client_explanations, save_path=overview_path, show_plot=False
            )
            if result:
                generated_plots['federated_overview'] = result
            
            # Individual client analysis
            for client_id, explanations in client_explanations.items():
                if not explanations:
                    continue
                
                # Latest explanation data
                latest_exp = explanations[-1]
                
                # Feature importance plot
                importance_path = f"{output_prefix}_client_{client_id}_importance.png"
                result = self.create_feature_importance_plot(
                    latest_exp.feature_importance,
                    title=f"Feature Importance - Client {client_id}",
                    save_path=importance_path,
                    show_plot=False
                )
                if result:
                    generated_plots[f'client_{client_id}_importance'] = result
                
                # Trust trends if available
                try:
                    trends = trust_engine.get_trust_trends(client_id)
                    if trends:
                        trends_path = f"{output_prefix}_client_{client_id}_trends.png"
                        result = self.create_trust_trends_plot(
                            trends, client_id,
                            save_path=trends_path, show_plot=False
                        )
                        if result:
                            generated_plots[f'client_{client_id}_trends'] = result
                except ValueError:
                    pass  # Skip if insufficient data
            
            logger.info(f"Generated {len(generated_plots)} visualization plots")
            return generated_plots
            
        except Exception as e:
            logger.error(f"Error generating visualization report: {e}")
            return generated_plots
