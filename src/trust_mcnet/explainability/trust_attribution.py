"""
Trust attribution module for federated learning with SHAP explanations.

This module provides functionality to analyze and attribute trust scores
based on client explanations and model predictions in a federated learning context.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClientExplanation:
    """
    Container for client-specific SHAP explanations and metadata.
    
    Attributes:
        client_id: Unique identifier for the client
        shap_values: SHAP explanation values
        predictions: Model predictions
        feature_importance: Feature importance scores
        base_value: Base prediction value
        confidence_score: Confidence in the predictions
        data_quality_score: Quality score of the client's data
        timestamp: When the explanation was generated
        metadata: Additional client-specific information
    """
    client_id: str
    shap_values: np.ndarray
    predictions: np.ndarray
    feature_importance: np.ndarray
    base_value: float
    confidence_score: float = 0.0
    data_quality_score: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrustMetrics:
    """
    Trust metrics computed from explanations and predictions.
    
    Attributes:
        explanation_consistency: How consistent are the explanations
        prediction_reliability: Reliability of predictions
        feature_stability: Stability of feature importance
        anomaly_detection_quality: Quality of anomaly detection
        overall_trust_score: Overall trust score (0-1)
        risk_assessment: Risk level (low, medium, high)
        recommendation: Recommended action
    """
    explanation_consistency: float
    prediction_reliability: float
    feature_stability: float
    anomaly_detection_quality: float
    overall_trust_score: float
    risk_assessment: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'explanation_consistency': self.explanation_consistency,
            'prediction_reliability': self.prediction_reliability,
            'feature_stability': self.feature_stability,
            'anomaly_detection_quality': self.anomaly_detection_quality,
            'overall_trust_score': self.overall_trust_score,
            'risk_assessment': self.risk_assessment,
            'recommendation': self.recommendation
        }


class TrustAttributionEngine:
    """
    Engine for computing trust scores based on SHAP explanations in federated learning.
    
    This class analyzes explanations from multiple clients to assess the trustworthiness
    of their contributions and the overall model performance.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        consistency_threshold: float = 0.8,
        reliability_threshold: float = 0.7,
        trust_threshold: float = 0.6
    ):
        """
        Initialize trust attribution engine.
        
        Args:
            feature_names: Names of input features
            consistency_threshold: Threshold for explanation consistency
            reliability_threshold: Threshold for prediction reliability
            trust_threshold: Threshold for overall trust
        """
        self.feature_names = feature_names or []
        self.consistency_threshold = consistency_threshold
        self.reliability_threshold = reliability_threshold
        self.trust_threshold = trust_threshold
        
        # Storage for client explanations
        self.client_explanations: Dict[str, List[ClientExplanation]] = defaultdict(list)
        self.global_feature_importance: Optional[np.ndarray] = None
        self.trust_history: List[Dict[str, Any]] = []
        
    def add_client_explanation(self, explanation: ClientExplanation) -> None:
        """
        Add a client explanation to the trust analysis.
        
        Args:
            explanation: Client explanation data
        """
        self.client_explanations[explanation.client_id].append(explanation)
        logger.info(f"Added explanation for client {explanation.client_id}")
        
        # Update global feature importance
        self._update_global_feature_importance()
    
    def compute_trust_metrics(
        self,
        client_id: Optional[str] = None,
        window_size: int = 10
    ) -> Union[TrustMetrics, Dict[str, TrustMetrics]]:
        """
        Compute trust metrics for a specific client or all clients.
        
        Args:
            client_id: Specific client ID, or None for all clients
            window_size: Number of recent explanations to consider
            
        Returns:
            Trust metrics for the client(s)
        """
        if client_id:
            return self._compute_client_trust_metrics(client_id, window_size)
        else:
            # Compute for all clients
            all_metrics = {}
            for cid in self.client_explanations.keys():
                all_metrics[cid] = self._compute_client_trust_metrics(cid, window_size)
            return all_metrics
    
    def _compute_client_trust_metrics(
        self,
        client_id: str,
        window_size: int
    ) -> TrustMetrics:
        """
        Compute trust metrics for a specific client.
        
        Args:
            client_id: Client identifier
            window_size: Number of recent explanations to consider
            
        Returns:
            Trust metrics for the client
        """
        if client_id not in self.client_explanations:
            raise ValueError(f"No explanations found for client {client_id}")
        
        explanations = self.client_explanations[client_id][-window_size:]
        
        if not explanations:
            raise ValueError(f"No explanations available for client {client_id}")
        
        # Extract data for analysis
        shap_values_list = [exp.shap_values for exp in explanations]
        predictions_list = [exp.predictions for exp in explanations]
        feature_importance_list = [exp.feature_importance for exp in explanations]
        
        # Compute individual metrics
        explanation_consistency = self._compute_explanation_consistency(shap_values_list)
        prediction_reliability = self._compute_prediction_reliability(predictions_list, explanations)
        feature_stability = self._compute_feature_stability(feature_importance_list)
        anomaly_quality = self._compute_anomaly_detection_quality(explanations)
        
        # Compute overall trust score
        overall_trust = self._compute_overall_trust_score(
            explanation_consistency,
            prediction_reliability,
            feature_stability,
            anomaly_quality
        )
        
        # Determine risk assessment and recommendation
        risk_assessment, recommendation = self._assess_risk_and_recommend(overall_trust)
        
        return TrustMetrics(
            explanation_consistency=explanation_consistency,
            prediction_reliability=prediction_reliability,
            feature_stability=feature_stability,
            anomaly_detection_quality=anomaly_quality,
            overall_trust_score=overall_trust,
            risk_assessment=risk_assessment,
            recommendation=recommendation
        )
    
    def _compute_explanation_consistency(
        self,
        shap_values_list: List[np.ndarray]
    ) -> float:
        """
        Compute consistency of SHAP explanations over time.
        
        Args:
            shap_values_list: List of SHAP values from different time points
            
        Returns:
            Consistency score (0-1)
        """
        if len(shap_values_list) < 2:
            return 1.0  # Single explanation is perfectly consistent
        
        # Compute pairwise correlations between explanation patterns
        correlations = []
        
        for i in range(len(shap_values_list)):
            for j in range(i + 1, len(shap_values_list)):
                # Average SHAP values across samples for each feature
                avg_shap_i = np.mean(np.abs(shap_values_list[i]), axis=0)
                avg_shap_j = np.mean(np.abs(shap_values_list[j]), axis=0)
                
                # Compute correlation
                if len(avg_shap_i) > 1 and len(avg_shap_j) > 1:
                    correlation = np.corrcoef(avg_shap_i, avg_shap_j)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
        
        if not correlations:
            return 0.5  # Neutral score if correlation can't be computed
        
        # Return average correlation
        return np.mean(correlations)
    
    def _compute_prediction_reliability(
        self,
        predictions_list: List[np.ndarray],
        explanations: List[ClientExplanation]
    ) -> float:
        """
        Compute reliability of predictions based on confidence scores.
        
        Args:
            predictions_list: List of predictions from different time points
            explanations: List of client explanations
            
        Returns:
            Reliability score (0-1)
        """
        # Use confidence scores if available
        confidence_scores = [exp.confidence_score for exp in explanations if exp.confidence_score > 0]
        
        if confidence_scores:
            return np.mean(confidence_scores)
        
        # Fallback: compute prediction stability
        if len(predictions_list) < 2:
            return 0.5  # Neutral score for single prediction
        
        # Compute standard deviation of anomaly rates
        anomaly_rates = [np.mean(pred) for pred in predictions_list]
        
        if len(set(anomaly_rates)) == 1:
            return 1.0  # Perfect stability
        
        # Lower standard deviation = higher reliability
        std_rate = np.std(anomaly_rates)
        reliability = max(0.0, 1.0 - std_rate * 2)  # Scale factor of 2
        
        return reliability
    
    def _compute_feature_stability(
        self,
        feature_importance_list: List[np.ndarray]
    ) -> float:
        """
        Compute stability of feature importance rankings.
        
        Args:
            feature_importance_list: List of feature importance arrays
            
        Returns:
            Stability score (0-1)
        """
        if len(feature_importance_list) < 2:
            return 1.0  # Single importance ranking is perfectly stable
        
        # Compute rank correlations between feature importance rankings
        rank_correlations = []
        
        for i in range(len(feature_importance_list)):
            for j in range(i + 1, len(feature_importance_list)):
                # Get rankings (higher importance = lower rank number)
                ranks_i = np.argsort(np.argsort(-feature_importance_list[i]))
                ranks_j = np.argsort(np.argsort(-feature_importance_list[j]))
                
                # Compute Spearman rank correlation
                correlation = np.corrcoef(ranks_i, ranks_j)[0, 1]
                if not np.isnan(correlation):
                    rank_correlations.append(correlation)
        
        if not rank_correlations:
            return 0.5  # Neutral score if correlation can't be computed
        
        return np.mean(rank_correlations)
    
    def _compute_anomaly_detection_quality(
        self,
        explanations: List[ClientExplanation]
    ) -> float:
        """
        Compute quality of anomaly detection based on data quality scores.
        
        Args:
            explanations: List of client explanations
            
        Returns:
            Quality score (0-1)
        """
        quality_scores = [exp.data_quality_score for exp in explanations if exp.data_quality_score > 0]
        
        if quality_scores:
            return np.mean(quality_scores)
        
        # Fallback: analyze SHAP value distributions
        shap_std_scores = []
        for exp in explanations:
            # Higher standard deviation in SHAP values suggests better discrimination
            std_values = np.std(exp.shap_values, axis=0)
            avg_std = np.mean(std_values)
            shap_std_scores.append(min(1.0, avg_std))  # Cap at 1.0
        
        return np.mean(shap_std_scores) if shap_std_scores else 0.5
    
    def _compute_overall_trust_score(
        self,
        consistency: float,
        reliability: float,
        stability: float,
        quality: float
    ) -> float:
        """
        Compute overall trust score from individual metrics.
        
        Args:
            consistency: Explanation consistency score
            reliability: Prediction reliability score
            stability: Feature stability score
            quality: Anomaly detection quality score
            
        Returns:
            Overall trust score (0-1)
        """
        # Weighted average with equal weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        scores = np.array([consistency, reliability, stability, quality])
        
        # Handle NaN values
        valid_mask = ~np.isnan(scores)
        if not np.any(valid_mask):
            return 0.5  # Neutral score if all metrics are invalid
        
        # Renormalize weights for valid scores
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)
        valid_scores = scores[valid_mask]
        
        return np.dot(valid_weights, valid_scores)
    
    def _assess_risk_and_recommend(
        self,
        trust_score: float
    ) -> Tuple[str, str]:
        """
        Assess risk level and provide recommendation based on trust score.
        
        Args:
            trust_score: Overall trust score
            
        Returns:
            Risk assessment and recommendation
        """
        if trust_score >= self.trust_threshold:
            return "low", "Client predictions are trustworthy. Continue monitoring."
        elif trust_score >= 0.4:
            return "medium", "Client shows moderate trust. Increase monitoring frequency."
        else:
            return "high", "Client shows low trust. Consider client removal or retraining."
    
    def _update_global_feature_importance(self) -> None:
        """Update global feature importance from all client explanations."""
        all_importance = []
        
        for client_explanations in self.client_explanations.values():
            for explanation in client_explanations:
                all_importance.append(explanation.feature_importance)
        
        if all_importance:
            self.global_feature_importance = np.mean(all_importance, axis=0)
    
    def get_top_trusted_clients(
        self,
        k: int = 5,
        window_size: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top K most trusted clients based on recent performance.
        
        Args:
            k: Number of top clients to return
            window_size: Number of recent explanations to consider
            
        Returns:
            List of (client_id, trust_score) tuples
        """
        client_trust_scores = []
        
        for client_id in self.client_explanations.keys():
            try:
                metrics = self._compute_client_trust_metrics(client_id, window_size)
                client_trust_scores.append((client_id, metrics.overall_trust_score))
            except ValueError:
                # Skip clients with insufficient data
                continue
        
        # Sort by trust score (descending)
        client_trust_scores.sort(key=lambda x: x[1], reverse=True)
        
        return client_trust_scores[:k]
    
    def get_trust_trends(
        self,
        client_id: str,
        window_size: int = 5
    ) -> Dict[str, List[float]]:
        """
        Get trust score trends for a specific client over time.
        
        Args:
            client_id: Client identifier
            window_size: Window size for computing rolling metrics
            
        Returns:
            Dictionary with trend data
        """
        if client_id not in self.client_explanations:
            raise ValueError(f"No explanations found for client {client_id}")
        
        explanations = self.client_explanations[client_id]
        
        if len(explanations) < window_size:
            logger.warning(f"Insufficient data for trends analysis for client {client_id}")
            return {}
        
        trends = {
            'timestamps': [],
            'trust_scores': [],
            'consistency_scores': [],
            'reliability_scores': []
        }
        
        # Compute rolling metrics
        for i in range(window_size - 1, len(explanations)):
            window_explanations = explanations[i - window_size + 1:i + 1]
            
            # Extract data for this window
            shap_values_list = [exp.shap_values for exp in window_explanations]
            predictions_list = [exp.predictions for exp in window_explanations]
            feature_importance_list = [exp.feature_importance for exp in window_explanations]
            
            # Compute metrics for this window
            consistency = self._compute_explanation_consistency(shap_values_list)
            reliability = self._compute_prediction_reliability(predictions_list, window_explanations)
            stability = self._compute_feature_stability(feature_importance_list)
            quality = self._compute_anomaly_detection_quality(window_explanations)
            
            trust_score = self._compute_overall_trust_score(consistency, reliability, stability, quality)
            
            # Store results
            trends['timestamps'].append(explanations[i].timestamp)
            trends['trust_scores'].append(trust_score)
            trends['consistency_scores'].append(consistency)
            trends['reliability_scores'].append(reliability)
        
        return trends
    
    def export_trust_report(
        self,
        filepath: str,
        include_trends: bool = True
    ) -> None:
        """
        Export comprehensive trust report to JSON file.
        
        Args:
            filepath: Path to save the report
            include_trends: Whether to include trend analysis
        """
        report = {
            'generation_time': datetime.now().isoformat(),
            'total_clients': len(self.client_explanations),
            'total_explanations': sum(len(exps) for exps in self.client_explanations.values()),
            'thresholds': {
                'consistency': self.consistency_threshold,
                'reliability': self.reliability_threshold,
                'trust': self.trust_threshold
            },
            'clients': {}
        }
        
        # Add client-specific metrics
        for client_id in self.client_explanations.keys():
            try:
                metrics = self._compute_client_trust_metrics(client_id, window_size=10)
                client_data = {
                    'trust_metrics': metrics.to_dict(),
                    'num_explanations': len(self.client_explanations[client_id]),
                    'last_update': self.client_explanations[client_id][-1].timestamp.isoformat()
                }
                
                # Add trends if requested
                if include_trends:
                    try:
                        trends = self.get_trust_trends(client_id)
                        if trends:
                            # Convert timestamps to ISO format for JSON serialization
                            trends['timestamps'] = [ts.isoformat() for ts in trends['timestamps']]
                            client_data['trends'] = trends
                    except ValueError:
                        pass  # Skip trends if insufficient data
                
                report['clients'][client_id] = client_data
                
            except ValueError:
                # Skip clients with insufficient data
                continue
        
        # Add global feature importance if available
        if self.global_feature_importance is not None:
            feature_importance_dict = {}
            for i, importance in enumerate(self.global_feature_importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_importance_dict[feature_name] = float(importance)
            
            report['global_feature_importance'] = feature_importance_dict
        
        # Add top trusted clients
        top_clients = self.get_top_trusted_clients(k=10)
        report['top_trusted_clients'] = [
            {'client_id': cid, 'trust_score': score}
            for cid, score in top_clients
        ]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Trust report exported to {filepath}")


class AlertingIntegration:
    """
    Integration module for embedding SHAP explanations into TRUST_MCNet alerting.
    
    This class provides methods to integrate trust attribution and SHAP explanations
    into the existing alerting mechanism.
    """
    
    def __init__(
        self,
        trust_engine: TrustAttributionEngine,
        alert_threshold: float = 0.5,
        explanation_top_k: int = 5
    ):
        """
        Initialize alerting integration.
        
        Args:
            trust_engine: Trust attribution engine
            alert_threshold: Threshold for triggering alerts
            explanation_top_k: Number of top features to include in alerts
        """
        self.trust_engine = trust_engine
        self.alert_threshold = alert_threshold
        self.explanation_top_k = explanation_top_k
        
    def generate_anomaly_alert(
        self,
        client_id: str,
        prediction: np.ndarray,
        shap_values: np.ndarray,
        feature_names: List[str],
        sample_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced anomaly alert with SHAP explanations.
        
        Args:
            client_id: Client identifier
            prediction: Model prediction (1 for anomaly, 0 for normal)
            shap_values: SHAP explanation values
            feature_names: Names of input features
            sample_data: Original sample data
            
        Returns:
            Enhanced alert with explanations
        """
        # Determine alert severity
        anomaly_rate = np.mean(prediction)
        
        if anomaly_rate < self.alert_threshold:
            alert_level = "info"
        elif anomaly_rate < 0.8:
            alert_level = "warning"
        else:
            alert_level = "critical"
        
        # Get top contributing features
        feature_contributions = self._get_top_feature_contributions(
            shap_values, feature_names, self.explanation_top_k
        )
        
        # Get client trust metrics if available
        trust_metrics = None
        try:
            trust_metrics = self.trust_engine.compute_trust_metrics(client_id)
            if isinstance(trust_metrics, dict):
                trust_metrics = trust_metrics.get(client_id)
        except (ValueError, KeyError):
            pass
        
        # Create alert
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_id': f"{client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'client_id': client_id,
            'alert_level': alert_level,
            'anomaly_rate': float(anomaly_rate),
            'num_samples': len(prediction),
            'top_features': feature_contributions,
            'trust_info': trust_metrics.to_dict() if trust_metrics else None,
            'explanation_method': 'SHAP',
            'recommended_action': self._get_recommended_action(alert_level, trust_metrics)
        }
        
        # Add sample data if provided
        if sample_data is not None:
            alert['sample_statistics'] = {
                'mean': np.mean(sample_data, axis=0).tolist(),
                'std': np.std(sample_data, axis=0).tolist(),
                'min': np.min(sample_data, axis=0).tolist(),
                'max': np.max(sample_data, axis=0).tolist()
            }
        
        return alert
    
    def _get_top_feature_contributions(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Get top K features contributing to the prediction.
        
        Args:
            shap_values: SHAP explanation values
            feature_names: Names of input features
            top_k: Number of top features to return
            
        Returns:
            List of top feature contributions
        """
        # Average SHAP values across samples
        avg_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get top K features
        top_indices = np.argsort(avg_shap)[-top_k:][::-1]
        
        contributions = []
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            contributions.append({
                'feature_name': feature_name,
                'feature_index': int(idx),
                'importance_score': float(avg_shap[idx]),
                'contribution_direction': 'positive' if np.mean(shap_values[:, idx]) > 0 else 'negative'
            })
        
        return contributions
    
    def _get_recommended_action(
        self,
        alert_level: str,
        trust_metrics: Optional[TrustMetrics]
    ) -> str:
        """
        Get recommended action based on alert level and trust metrics.
        
        Args:
            alert_level: Level of the alert
            trust_metrics: Trust metrics for the client
            
        Returns:
            Recommended action string
        """
        if trust_metrics and trust_metrics.overall_trust_score < 0.4:
            return "High-priority investigation required. Consider client isolation."
        
        if alert_level == "critical":
            return "Immediate investigation required. Potential security threat."
        elif alert_level == "warning":
            return "Monitor closely. Investigate if pattern persists."
        else:
            return "Continue normal monitoring."
    
    def format_alert_message(self, alert: Dict[str, Any]) -> str:
        """
        Format alert for human-readable output.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Formatted alert message
        """
        message = f"""
🚨 TRUST_MCNet Anomaly Alert 🚨

Alert ID: {alert['alert_id']}
Timestamp: {alert['timestamp']}
Client: {alert['client_id']}
Level: {alert['alert_level'].upper()}
Anomaly Rate: {alert['anomaly_rate']:.2%}

Top Contributing Features:
"""
        
        for i, feature in enumerate(alert['top_features'], 1):
            direction = "↑" if feature['contribution_direction'] == 'positive' else "↓"
            message += f"  {i}. {feature['feature_name']} {direction} (score: {feature['importance_score']:.3f})\n"
        
        if alert['trust_info']:
            trust_score = alert['trust_info']['overall_trust_score']
            risk = alert['trust_info']['risk_assessment']
            message += f"\nClient Trust: {trust_score:.2f} (Risk: {risk})"
        
        message += f"\nRecommended Action: {alert['recommended_action']}"
        
        return message
