#!/usr/bin/env python3
"""
Demonstration script for TRUST_MCNet Enhanced SHAP Integration.

This script demonstrates how to use the enhanced explainability module to:
1. Train traditional ML models (XGBoost, RandomForest, IsolationForest)
2. Generate SHAP explanations for anomaly detection
3. Compute trust metrics for federated learning clients
4. Create comprehensive visualizations
5. Integrate with TRUST_MCNet alerting system

Usage:
    python demo_enhanced_explainability.py [--dataset <path>] [--model <type>] [--output <dir>]

Example:
    python demo_enhanced_explainability.py --dataset data/iot_traffic.csv --model xgboost --output results/
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check if we can import our enhanced explainability module
try:
    from explainability import (
        EnhancedSHAPExplainer,
        AnomalyExplanationPipeline,
        TrustAttributionEngine,
        ClientExplanation,
        TrustMetrics,
        AlertingIntegration,
        SHAPVisualizationManager,
        XGBoostExplainer,
        RandomForestExplainer,
        IsolationForestExplainer,
        AVAILABILITY_INFO,
        check_dependencies,
        get_installation_commands
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced explainability module not available: {e}")
    EXPLAINABILITY_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_explainability.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_synthetic_iot_data(
    n_samples: int = 1000,
    n_features: int = 20,
    anomaly_rate: float = 0.1,
    random_state: int = 42
) -> tuple:
    """
    Generate synthetic IoT network traffic data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        anomaly_rate: Proportion of anomalous samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, feature_names) where X is features, y is labels
    """
    np.random.seed(random_state)
    
    # Feature names representing network traffic characteristics
    feature_names = [
        'packet_size_mean', 'packet_size_std', 'packet_count',
        'flow_duration', 'bytes_per_second', 'packets_per_second',
        'tcp_flags_count', 'unique_ports', 'connection_rate',
        'protocol_distribution', 'payload_entropy', 'header_length_mean',
        'inter_arrival_time_mean', 'inter_arrival_time_std',
        'connection_state_changes', 'retransmission_rate',
        'window_size_mean', 'fragmentation_rate', 'dns_query_rate',
        'error_rate'
    ]
    
    # Adjust feature names if needed
    if n_features != len(feature_names):
        if n_features < len(feature_names):
            feature_names = feature_names[:n_features]
        else:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
    
    # Generate normal traffic data
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomalous = n_samples - n_normal
    
    # Normal traffic: moderate values with some correlation structure
    normal_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 0.3,
        cov=np.eye(n_features) * 0.1 + np.ones((n_features, n_features)) * 0.02,
        size=n_normal
    )
    
    # Anomalous traffic: extreme values and different patterns
    anomalous_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 0.8,
        cov=np.eye(n_features) * 0.3,
        size=n_anomalous
    )
    
    # Add some structured anomalies
    anomalous_data[:n_anomalous//2, :5] *= 3  # High packet-related features
    anomalous_data[n_anomalous//2:, -5:] *= 0.1  # Low connection-related features
    
    # Combine data
    X = np.vstack([normal_data, anomalous_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalous)])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Ensure non-negative values for some features
    X = np.abs(X)
    
    logger.info(f"Generated synthetic IoT data: {n_samples} samples, {n_features} features")
    logger.info(f"Anomaly rate: {np.mean(y):.2%}")
    
    return X, y, feature_names


def create_client_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    random_state: int = 42
) -> Dict[str, tuple]:
    """
    Split data among federated learning clients.
    
    Args:
        X: Feature data
        y: Labels
        n_clients: Number of clients
        random_state: Random seed
        
    Returns:
        Dictionary mapping client IDs to (X_client, y_client) tuples
    """
    np.random.seed(random_state)
    
    # Create client data with different distributions
    client_data = {}
    n_samples = len(X)
    
    # Generate unequal client data sizes
    client_sizes = np.random.dirichlet(np.ones(n_clients)) * n_samples
    client_sizes = client_sizes.astype(int)
    client_sizes[-1] += n_samples - client_sizes.sum()  # Adjust for rounding
    
    start_idx = 0
    for i in range(n_clients):
        client_id = f"client_{i+1:02d}"
        end_idx = start_idx + client_sizes[i]
        
        # Get client data slice
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        
        # Add some client-specific bias
        bias_factor = np.random.normal(1.0, 0.1)
        X_client = X_client * bias_factor
        
        client_data[client_id] = (X_client, y_client)
        start_idx = end_idx
        
        logger.info(f"Client {client_id}: {len(X_client)} samples, "
                   f"anomaly rate: {np.mean(y_client):.2%}")
    
    return client_data


def demonstrate_model_training_and_explanation(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Demonstrate training a model and generating SHAP explanations.
    
    Args:
        model_type: Type of model ('xgboost', 'randomforest', 'isolationforest')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results and explanations
    """
    logger.info(f"Training {model_type} model...")
    
    # Create model explainer based on type
    if model_type == 'xgboost':
        if XGBoostExplainer is None:
            raise ImportError("XGBoost not available")
        explainer = XGBoostExplainer(feature_names=feature_names)
    elif model_type == 'randomforest':
        if RandomForestExplainer is None:
            raise ImportError("Random Forest not available")
        explainer = RandomForestExplainer(feature_names=feature_names)
    elif model_type == 'isolationforest':
        if IsolationForestExplainer is None:
            raise ImportError("Isolation Forest not available")
        explainer = IsolationForestExplainer(feature_names=feature_names)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    training_results = explainer.train(X_train, y_train, X_test, y_test)
    logger.info(f"Training completed. Accuracy: {training_results.get('val_accuracy', 'N/A')}")
    
    # Generate predictions
    predictions = explainer.predict(X_test)
    logger.info(f"Test anomaly rate: {np.mean(predictions):.2%}")
    
    # Generate SHAP explanations
    logger.info("Generating SHAP explanations...")
    
    if model_type == 'isolationforest':
        # Isolation Forest needs background data for KernelExplainer
        explanation_results = explainer.explain_predictions(
            X_test[:100],  # Limit samples for performance
            background_data=X_train[:50],  # Background data
            max_samples=50
        )
    else:
        # Tree models use TreeExplainer (faster)
        explanation_results = explainer.explain_predictions(
            X_test[:100],  # Limit samples for performance
            max_samples=100
        )
    
    # Save model
    model_path = output_dir / f"{model_type}_model.pkl"
    explainer.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Combine results
    results = {
        'model_type': model_type,
        'training_results': training_results,
        'predictions': predictions,
        'explanation_results': explanation_results,
        'model_path': str(model_path)
    }
    
    return results


def demonstrate_trust_attribution(
    client_explanations_data: Dict[str, List[Dict[str, Any]]],
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Demonstrate trust attribution for federated learning clients.
    
    Args:
        client_explanations_data: Dictionary of client explanation data
        feature_names: Feature names
        output_dir: Output directory
        
    Returns:
        Trust attribution results
    """
    logger.info("Demonstrating trust attribution for federated learning...")
    
    # Create trust attribution engine
    trust_engine = TrustAttributionEngine(
        feature_names=feature_names,
        consistency_threshold=0.8,
        reliability_threshold=0.7,
        trust_threshold=0.6
    )
    
    # Convert data to ClientExplanation objects and add to engine
    for client_id, explanations_list in client_explanations_data.items():
        for exp_data in explanations_list:
            explanation = ClientExplanation(
                client_id=client_id,
                shap_values=exp_data['shap_values'],
                predictions=exp_data['predictions'],
                feature_importance=exp_data['feature_importance'],
                base_value=exp_data['base_value'],
                confidence_score=np.random.uniform(0.5, 0.9),  # Simulated confidence
                data_quality_score=np.random.uniform(0.6, 0.95)  # Simulated quality
            )
            trust_engine.add_client_explanation(explanation)
    
    # Compute trust metrics for all clients
    all_trust_metrics = trust_engine.compute_trust_metrics()
    
    # Display trust metrics
    logger.info("Trust metrics computed:")
    for client_id, metrics in all_trust_metrics.items():
        logger.info(f"  {client_id}: Trust={metrics.overall_trust_score:.3f}, "
                   f"Risk={metrics.risk_assessment}")
    
    # Get top trusted clients
    top_clients = trust_engine.get_top_trusted_clients(k=3)
    logger.info(f"Top trusted clients: {top_clients}")
    
    # Export trust report
    report_path = output_dir / "trust_report.json"
    trust_engine.export_trust_report(str(report_path))
    logger.info(f"Trust report exported to {report_path}")
    
    return {
        'trust_metrics': {cid: metrics.to_dict() for cid, metrics in all_trust_metrics.items()},
        'top_clients': top_clients,
        'report_path': str(report_path)
    }


def demonstrate_alerting_integration(
    explanation_results: Dict[str, Any],
    trust_engine: Any,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Demonstrate integration with TRUST_MCNet alerting system.
    
    Args:
        explanation_results: SHAP explanation results
        trust_engine: Trust attribution engine
        feature_names: Feature names
        output_dir: Output directory
        
    Returns:
        Alerting integration results
    """
    logger.info("Demonstrating alerting integration...")
    
    # Create alerting integration
    alerting = AlertingIntegration(
        trust_engine=trust_engine,
        alert_threshold=0.3,
        explanation_top_k=5
    )
    
    # Generate sample alert
    client_id = "client_01"
    predictions = explanation_results['predictions'][:10]  # Sample predictions
    shap_values = explanation_results['shap_values'][:10]  # Sample SHAP values
    
    # Generate enhanced alert
    alert = alerting.generate_anomaly_alert(
        client_id=client_id,
        prediction=predictions,
        shap_values=shap_values,
        feature_names=feature_names
    )
    
    # Format alert message
    alert_message = alerting.format_alert_message(alert)
    
    # Save alert
    alert_path = output_dir / f"alert_{alert['alert_id']}.json"
    with open(alert_path, 'w') as f:
        json.dump(alert, f, indent=2, default=str)
    
    # Save formatted message
    message_path = output_dir / f"alert_message_{alert['alert_id']}.txt"
    with open(message_path, 'w') as f:
        f.write(alert_message)
    
    logger.info(f"Alert generated and saved to {alert_path}")
    logger.info("Alert message:")
    logger.info(alert_message)
    
    return {
        'alert': alert,
        'alert_message': alert_message,
        'alert_path': str(alert_path),
        'message_path': str(message_path)
    }


def demonstrate_visualizations(
    explanation_results: Dict[str, Any],
    trust_metrics: Dict[str, Any],
    client_explanations_data: Dict[str, List[Dict[str, Any]]],
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, str]:
    """
    Demonstrate comprehensive visualization capabilities.
    
    Args:
        explanation_results: SHAP explanation results
        trust_metrics: Trust metrics
        client_explanations_data: Client explanations data
        feature_names: Feature names
        output_dir: Output directory
        
    Returns:
        Dictionary of generated visualization paths
    """
    logger.info("Generating comprehensive visualizations...")
    
    if SHAPVisualizationManager is None:
        logger.warning("Visualization manager not available (matplotlib missing)")
        return {}
    
    # Create visualization manager
    viz_manager = SHAPVisualizationManager(
        feature_names=feature_names,
        output_dir=str(output_dir / "visualizations")
    )
    
    generated_plots = {}
    
    try:
        # Feature importance plot
        importance_path = viz_manager.create_feature_importance_plot(
            explanation_results['feature_importance'],
            title="SHAP Feature Importance",
            save_path="feature_importance.png",
            show_plot=False
        )
        if importance_path:
            generated_plots['feature_importance'] = importance_path
        
        # Trust metrics dashboard (if we have proper trust metrics objects)
        if trust_metrics and isinstance(list(trust_metrics.values())[0], dict):
            # Convert dict format to TrustMetrics objects for visualization
            from explainability.trust_attribution import TrustMetrics
            trust_objects = {}
            for client_id, metrics_dict in trust_metrics.items():
                trust_objects[client_id] = TrustMetrics(
                    explanation_consistency=metrics_dict.get('explanation_consistency', 0.5),
                    prediction_reliability=metrics_dict.get('prediction_reliability', 0.5),
                    feature_stability=metrics_dict.get('feature_stability', 0.5),
                    anomaly_detection_quality=metrics_dict.get('anomaly_detection_quality', 0.5),
                    overall_trust_score=metrics_dict.get('overall_trust_score', 0.5),
                    risk_assessment=metrics_dict.get('risk_assessment', 'medium'),
                    recommendation=metrics_dict.get('recommendation', 'Monitor closely')
                )
            
            dashboard_path = viz_manager.create_trust_metrics_dashboard(
                trust_objects,
                save_path="trust_dashboard.png",
                show_plot=False
            )
            if dashboard_path:
                generated_plots['trust_dashboard'] = dashboard_path
        
        logger.info(f"Generated {len(generated_plots)} visualization plots")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    return generated_plots


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate TRUST_MCNet Enhanced SHAP Integration"
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to dataset (if not provided, synthetic data will be generated)'
    )
    parser.add_argument(
        '--model', type=str, choices=['xgboost', 'randomforest', 'isolationforest'],
        default='xgboost', help='Model type to demonstrate'
    )
    parser.add_argument(
        '--output', type=str, default='demo_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--samples', type=int, default=2000,
        help='Number of samples for synthetic data'
    )
    parser.add_argument(
        '--clients', type=int, default=5,
        help='Number of federated learning clients'
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not EXPLAINABILITY_AVAILABLE:
        print("Enhanced explainability module not available.")
        print("Please ensure the explainability module is properly installed.")
        return 1
    
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    if not deps['all_available']:
        logger.warning("Some dependencies are missing:")
        for dep, cmd in deps['missing']:
            logger.warning(f"  {dep}: {cmd}")
        logger.info("Available dependencies: " + ", ".join(deps['available']))
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting TRUST_MCNet SHAP Integration Demo")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Generate or load data
        if args.dataset:
            logger.info(f"Loading dataset from {args.dataset}")
            # TODO: Implement dataset loading logic
            raise NotImplementedError("Dataset loading not implemented yet")
        else:
            logger.info("Generating synthetic IoT traffic data...")
            X, y, feature_names = generate_synthetic_iot_data(
                n_samples=args.samples,
                n_features=20,
                anomaly_rate=0.15
            )
        
        # Split into train/test
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Step 2: Create federated client data
        client_data = create_client_data_splits(X_train, y_train, n_clients=args.clients)
        
        # Step 3: Demonstrate model training and explanation
        main_results = demonstrate_model_training_and_explanation(
            args.model, X_train, y_train, X_test, y_test, feature_names, output_dir
        )
        
        # Step 4: Simulate client explanations for federated learning
        logger.info("Simulating client explanations for federated learning demo...")
        client_explanations_data = {}
        
        for client_id, (X_client, y_client) in client_data.items():
            # Simulate multiple explanation rounds per client
            client_explanations_data[client_id] = []
            
            for round_idx in range(3):  # 3 rounds of explanations
                # Sample some data for explanation
                sample_size = min(50, len(X_client))
                sample_indices = np.random.choice(len(X_client), sample_size, replace=False)
                
                # Simulate SHAP values and predictions
                simulated_shap = np.random.normal(0, 0.1, (sample_size, len(feature_names)))
                simulated_predictions = np.random.binomial(1, 0.1, sample_size)
                simulated_importance = np.random.exponential(0.1, len(feature_names))
                
                client_explanations_data[client_id].append({
                    'shap_values': simulated_shap,
                    'predictions': simulated_predictions,
                    'feature_importance': simulated_importance,
                    'base_value': 0.1
                })
        
        # Step 5: Demonstrate trust attribution
        trust_results = demonstrate_trust_attribution(
            client_explanations_data, feature_names, output_dir
        )
        
        # Step 6: Demonstrate alerting integration
        alerting_results = demonstrate_alerting_integration(
            main_results['explanation_results'],
            None,  # We'll create a dummy trust engine in the function
            feature_names,
            output_dir
        )
        
        # Step 7: Demonstrate visualizations
        viz_results = demonstrate_visualizations(
            main_results['explanation_results'],
            trust_results['trust_metrics'],
            client_explanations_data,
            feature_names,
            output_dir
        )
        
        # Step 8: Create summary report
        summary = {
            'demo_timestamp': datetime.now().isoformat(),
            'model_type': args.model,
            'dataset_info': {
                'samples': len(X),
                'features': len(feature_names),
                'anomaly_rate': float(np.mean(y))
            },
            'training_results': main_results['training_results'],
            'trust_analysis': {
                'total_clients': len(client_data),
                'top_trusted_clients': trust_results['top_clients']
            },
            'generated_files': {
                'model_path': main_results['model_path'],
                'trust_report': trust_results['report_path'],
                'alert_path': alerting_results['alert_path'],
                'visualizations': viz_results
            },
            'dependencies_status': deps
        }
        
        # Save summary
        summary_path = output_dir / "demo_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Demo completed successfully!")
        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"Generated {len(viz_results)} visualizations")
        
        # Display key results
        print("\n" + "="*60)
        print("TRUST_MCNet Enhanced SHAP Integration Demo Results")
        print("="*60)
        print(f"Model Type: {args.model}")
        print(f"Test Accuracy: {main_results['training_results'].get('val_accuracy', 'N/A')}")
        print(f"Anomaly Detection Rate: {np.mean(main_results['predictions']):.2%}")
        print(f"Top Trusted Clients: {trust_results['top_clients'][:3]}")
        print(f"Alert Generated: {alerting_results['alert']['alert_level'].upper()}")
        print(f"Visualizations: {len(viz_results)} plots generated")
        print(f"Output Directory: {output_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
