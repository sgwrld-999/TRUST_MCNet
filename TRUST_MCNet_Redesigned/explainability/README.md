# Enhanced SHAP Explainability for TRUST_MCNet

This module provides comprehensive **SHAP (SHapley Additive exPlanations)** integration for the TRUST_MCNet federated learning framework, enabling interpretable anomaly detection for encrypted traffic analysis.

## 🎯 Features

### Core Explainability
- **Multi-Model SHAP Support**: PyTorch neural networks, XGBoost, RandomForest, IsolationForest
- **Optimized Performance**: TreeExplainer for tree models, KernelExplainer for others
- **Selective Explanations**: Generate explanations only for anomalous samples to reduce computational overhead
- **Caching System**: Intelligent caching of SHAP explainers and background data

### Federated Learning Integration
- **Trust Attribution**: Analyze client trustworthiness based on explanation consistency
- **Client Monitoring**: Track explanation patterns across federated learning rounds
- **Risk Assessment**: Automated risk scoring for client contributions
- **Global Feature Importance**: Aggregate feature importance across all clients

### Visualization Suite
- **Static Plots**: matplotlib/seaborn-based plots for reports
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Trust Metrics**: Comprehensive client trust dashboards
- **Federated Overview**: Multi-client analysis and comparison

### Alerting Integration
- **Enhanced Alerts**: SHAP-enriched anomaly alerts with feature explanations
- **Trust-Aware Alerting**: Alert severity based on client trust scores
- **Top-K Features**: Highlight most important features contributing to anomalies
- **Actionable Recommendations**: Automated response suggestions

## 📦 Installation

### Prerequisites

Install the required dependencies:

```bash
# Core ML and explainability
pip install shap scikit-learn xgboost

# Deep learning (if using PyTorch models)
pip install torch torchvision

# Visualization
pip install matplotlib seaborn plotly

# Data manipulation
pip install pandas numpy
```

### Check Dependencies

```python
from explainability import check_dependencies, get_installation_commands

# Check what's available
deps = check_dependencies()
print("Available:", deps['available'])
print("Missing:", deps['missing'])

# Get installation commands for missing dependencies
if not deps['all_available']:
    commands = get_installation_commands()
    for cmd in commands:
        print(f"Run: {cmd}")
```

## 🚀 Quick Start

### 1. Basic SHAP Explanation

```python
from explainability import EnhancedSHAPExplainer
import numpy as np

# Sample data (replace with your IoT traffic features)
X_train = np.random.randn(1000, 20)  # 1000 samples, 20 features
X_test = np.random.randn(200, 20)    # 200 test samples
y_train = np.random.binomial(1, 0.1, 1000)  # 10% anomaly rate

# Create explainer
explainer = EnhancedSHAPExplainer(
    model_type='xgboost',
    feature_names=[f'network_feature_{i}' for i in range(20)]
)

# Train and explain
explainer.fit(X_train, y_train)
explanations = explainer.explain(X_test, explain_all=False)  # Only anomalies

# Get top contributing features
top_features = explainer.get_top_features(explanations, k=5)
print("Top anomaly indicators:", top_features)
```

### 2. Federated Learning Trust Attribution

```python
from explainability import TrustAttributionEngine, ClientExplanation
from datetime import datetime

# Create trust engine
trust_engine = TrustAttributionEngine(
    feature_names=[f'feature_{i}' for i in range(20)],
    trust_threshold=0.6
)

# Add client explanations (from federated rounds)
for client_id in ['client_01', 'client_02', 'client_03']:
    explanation = ClientExplanation(
        client_id=client_id,
        shap_values=explanations['shap_values'],
        predictions=explanations['predictions'],
        feature_importance=explanations['feature_importance'],
        base_value=explanations['base_value'],
        confidence_score=0.85,
        data_quality_score=0.90
    )
    trust_engine.add_client_explanation(explanation)

# Compute trust metrics
trust_metrics = trust_engine.compute_trust_metrics()
for client_id, metrics in trust_metrics.items():
    print(f"{client_id}: Trust={metrics.overall_trust_score:.3f}, "
          f"Risk={metrics.risk_assessment}")

# Get most trusted clients
top_clients = trust_engine.get_top_trusted_clients(k=3)
print("Most trusted clients:", top_clients)
```

### 3. Enhanced Alerting

```python
from explainability import AlertingIntegration

# Create alerting integration
alerting = AlertingIntegration(
    trust_engine=trust_engine,
    alert_threshold=0.3,
    explanation_top_k=5
)

# Generate enhanced alert
alert = alerting.generate_anomaly_alert(
    client_id='client_01',
    prediction=predictions,
    shap_values=shap_values,
    feature_names=feature_names
)

# Format for human consumption
alert_message = alerting.format_alert_message(alert)
print(alert_message)
```

### 4. Comprehensive Visualization

```python
from explainability import SHAPVisualizationManager

# Create visualization manager
viz_manager = SHAPVisualizationManager(
    feature_names=feature_names,
    output_dir="visualizations"
)

# Generate various plots
viz_manager.create_feature_importance_plot(
    feature_importance, 
    save_path="importance.png"
)

viz_manager.create_trust_metrics_dashboard(
    trust_metrics,
    save_path="trust_dashboard.png"
)

# Generate comprehensive report
generated_plots = viz_manager.generate_visualization_report(
    client_explanations, 
    trust_engine,
    output_prefix="trust_mcnet_analysis"
)
```

## 🔧 Advanced Usage

### Traditional ML Model Wrappers

For XGBoost, RandomForest, and IsolationForest models:

```python
from explainability import XGBoostExplainer, RandomForestExplainer, IsolationForestExplainer

# XGBoost with optimized TreeExplainer
xgb_explainer = XGBoostExplainer(
    model_params={'n_estimators': 100, 'max_depth': 6},
    feature_names=feature_names
)
xgb_explainer.train(X_train, y_train, X_val, y_val)
xgb_explanations = xgb_explainer.explain_predictions(X_test)

# Isolation Forest for unsupervised anomaly detection  
iso_explainer = IsolationForestExplainer(
    model_params={'contamination': 0.1, 'n_estimators': 100}
)
iso_explainer.train(X_train, y_train)  # Labels used only for evaluation
iso_explanations = iso_explainer.explain_predictions(X_test, X_train[:100])
```

### Performance Optimization

```python
# Use caching for repeated explanations
explainer = EnhancedSHAPExplainer(
    model_type='xgboost',
    enable_caching=True,
    cache_size=1000,
    performance_mode=True  # Optimize for speed over precision
)

# Explain only anomalies to reduce computation
explanations = explainer.explain(
    X_test, 
    explain_all=False,           # Only explain anomalies
    max_samples_per_class=50,    # Limit samples per class
    background_size=100          # Smaller background dataset
)
```

### Trust Trends Analysis

```python
# Analyze trust evolution over time
trends = trust_engine.get_trust_trends('client_01', window_size=10)

# Visualize trends
viz_manager.create_trust_trends_plot(
    trends, 
    'client_01',
    save_path="client_01_trends.png"
)

# Export comprehensive trust report
trust_engine.export_trust_report(
    "comprehensive_trust_report.json",
    include_trends=True
)
```

## 📊 Integration with TRUST_MCNet Alerting

The enhanced explainability module seamlessly integrates with TRUST_MCNet's alerting system:

### Alert Enhancement Strategy

1. **Feature-Level Explanations**: Each alert includes top-K features contributing to the anomaly detection
2. **Trust-Aware Severity**: Alert levels are adjusted based on client trust scores
3. **Actionable Insights**: Recommendations are provided based on explanation patterns
4. **Historical Context**: Trends analysis informs alert prioritization

### Example Alert Output

```
🚨 TRUST_MCNet Anomaly Alert 🚨

Alert ID: client_01_20241205_143022
Timestamp: 2024-12-05T14:30:22
Client: client_01
Level: WARNING
Anomaly Rate: 15.30%

Top Contributing Features:
  1. packet_size_mean ↑ (score: 0.245)
  2. connection_rate ↑ (score: 0.189)
  3. payload_entropy ↓ (score: 0.156)
  4. bytes_per_second ↑ (score: 0.134)
  5. protocol_distribution ↑ (score: 0.098)

Client Trust: 0.72 (Risk: low)
Recommended Action: Monitor closely. Investigate if pattern persists.
```

### Embedding in Existing Alerting Flow

```python
def enhanced_anomaly_detection_pipeline(client_data, client_id):
    """Enhanced anomaly detection with SHAP explanations."""
    
    # 1. Standard anomaly detection
    predictions = model.predict(client_data)
    
    # 2. Generate explanations for anomalies only
    if np.any(predictions):
        explanations = explainer.explain(
            client_data[predictions == 1],
            explain_all=False
        )
        
        # 3. Update trust metrics
        trust_engine.add_client_explanation(
            ClientExplanation(
                client_id=client_id,
                shap_values=explanations['shap_values'],
                predictions=predictions[predictions == 1],
                feature_importance=explanations['feature_importance'],
                base_value=explanations['base_value']
            )
        )
        
        # 4. Generate enhanced alert
        alert = alerting.generate_anomaly_alert(
            client_id=client_id,
            prediction=predictions,
            shap_values=explanations['shap_values'],
            feature_names=feature_names,
            sample_data=client_data
        )
        
        # 5. Send alert through existing TRUST_MCNet channels
        send_enhanced_alert(alert)
```

## 🎨 Visualization Gallery

The module generates various visualization types:

### Static Visualizations (matplotlib/seaborn)
- **Feature Importance Bar Charts**: Top-K most important features
- **SHAP Summary Plots**: Feature effects across all samples  
- **Waterfall Plots**: Step-by-step explanation for individual predictions
- **Trust Metrics Dashboard**: Multi-client trust comparison
- **Federated Overview**: Client contributions and data quality

### Interactive Visualizations (plotly)
- **Interactive SHAP Dashboard**: Drill-down explanations
- **Trust Trends**: Time-series client trust evolution
- **Feature Correlation Heatmaps**: SHAP value correlations
- **Client Comparison Tools**: Side-by-side analysis

## 🧪 Running the Demo

A comprehensive demonstration script showcases all features:

```bash
# Run with synthetic data and XGBoost
python demo_enhanced_explainability.py \
    --model xgboost \
    --samples 2000 \
    --clients 5 \
    --output demo_results/

# Use different model types
python demo_enhanced_explainability.py --model randomforest
python demo_enhanced_explainability.py --model isolationforest

# Check what will be generated
python demo_enhanced_explainability.py --help
```

### Demo Outputs

The demo generates:
- `demo_results/xgboost_model.pkl`: Trained model
- `demo_results/trust_report.json`: Comprehensive trust analysis
- `demo_results/alert_*.json`: Sample enhanced alerts
- `demo_results/visualizations/`: All generated plots
- `demo_results/demo_summary.json`: Complete results summary

## 🔍 Technical Details

### Model Support

| Model Type | SHAP Explainer | Performance | Use Case |
|------------|----------------|-------------|----------|
| XGBoost | TreeExplainer | ⭐⭐⭐⭐⭐ | High-speed tree explanations |
| RandomForest | TreeExplainer | ⭐⭐⭐⭐⭐ | Robust ensemble explanations |
| IsolationForest | KernelExplainer | ⭐⭐⭐ | Unsupervised anomaly detection |
| PyTorch Neural Networks | DeepExplainer | ⭐⭐⭐⭐ | Deep learning explanations |

### Performance Optimizations

1. **TreeExplainer Usage**: Fast exact SHAP values for tree-based models
2. **Selective Explanation**: Only explain anomalous samples by default
3. **Background Data Sampling**: Intelligent sampling for KernelExplainer
4. **Caching System**: Cache explainers and background data
5. **Batch Processing**: Process multiple samples efficiently

### Trust Attribution Metrics

- **Explanation Consistency**: Correlation of SHAP patterns over time
- **Prediction Reliability**: Stability of anomaly detection rates  
- **Feature Stability**: Consistency of feature importance rankings
- **Anomaly Detection Quality**: Quality of anomaly vs normal discrimination
- **Overall Trust Score**: Weighted combination of all metrics

## 🤝 Contributing

To extend the module:

1. **Add New Model Types**: Inherit from `MLModelWrapper`
2. **Custom Trust Metrics**: Extend `TrustAttributionEngine`  
3. **Additional Visualizations**: Add methods to `SHAPVisualizationManager`
4. **Enhanced Alerting**: Customize `AlertingIntegration`

### Example: Adding Support for New Model

```python
from explainability.model_wrappers import MLModelWrapper

class CustomModelExplainer(MLModelWrapper):
    def _create_model(self):
        return YourCustomModel(**self.model_params)
    
    def _get_model_type(self):
        return "custom_model"
    
    def create_shap_explainer(self, background_data):
        # Implement appropriate SHAP explainer
        return shap.Explainer(self.model, background_data)
```

## 📝 License

This enhanced explainability module is part of the TRUST_MCNet project and follows the same licensing terms.

## 🙋‍♀️ Support

For issues or questions:
1. Check the dependency status with `check_dependencies()`
2. Review the demo script for usage examples
3. Examine generated logs for debugging information
4. Consult the comprehensive visualization outputs
