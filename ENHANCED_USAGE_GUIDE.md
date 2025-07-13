# Enhanced TRUST_MCNet Usage Examples

This document provides comprehensive examples for using the new adaptive trust strategies and real-time monitoring features.

## Quick Start with Enhanced Features

### 1. Adaptive Trust Strategy

The adaptive trust strategy automatically adjusts trust thresholds based on performance trends.

```bash
# Start adaptive server with monitoring
python main.py flower_server --num-rounds 20 --verbose
```

Use the enhanced configuration:
```bash
cp config/enhanced_federated.yaml config/my_adaptive_config.yaml
# Edit my_adaptive_config.yaml as needed
python main.py flower_server --config config/my_adaptive_config.yaml
```

### 2. Real-time Trust Monitoring

Enable the monitoring dashboard to track trust evolution in real-time:

```yaml
# In config/config.yaml
federated:
  enable_monitoring: true
  dashboard_output_dir: "trust_dashboard"
  dashboard_update_interval: 30
```

The dashboard generates:
- `trust_dashboard/dashboard.png` - Combined real-time dashboard
- `trust_dashboard/trust_evolution.png` - Trust score evolution over time
- `trust_dashboard/trust_distribution.png` - Current trust distribution
- `trust_dashboard/performance_vs_trust.png` - Performance correlation analysis
- `trust_dashboard/adaptation_timeline.png` - Threshold adaptation events
- `trust_dashboard/trust_metrics.json` - Exportable metrics data
- `trust_dashboard/status.json` - Current system status

## Configuration Options

### Adaptive Trust Strategy Parameters

```yaml
federated:
  use_adaptive_strategy: true
  target_accuracy: 0.85          # Target accuracy for adaptation
  adaptation_rate: 0.05          # Rate of threshold adjustment
  max_threshold: 0.9             # Maximum trust threshold
  min_threshold: 0.3             # Minimum trust threshold
  performance_window: 5          # Rounds to consider for trends
  adaptation_patience: 3         # Rounds to wait before adapting
```

### Monitoring Dashboard Parameters

```yaml
federated:
  enable_monitoring: true
  dashboard_output_dir: "trust_dashboard"
  dashboard_update_interval: 30  # Update interval in seconds
```

## Example Scenarios

### Scenario 1: High-Performance Requirements

For scenarios requiring high accuracy with strict trust requirements:

```yaml
federated:
  use_adaptive_strategy: true
  target_accuracy: 0.90
  adaptation_rate: 0.03
  max_threshold: 0.95
  min_threshold: 0.5
  performance_window: 3
  adaptation_patience: 2
```

### Scenario 2: Inclusive Learning

For scenarios prioritizing client inclusion while maintaining quality:

```yaml
federated:
  use_adaptive_strategy: true
  target_accuracy: 0.75
  adaptation_rate: 0.07
  max_threshold: 0.8
  min_threshold: 0.2
  performance_window: 7
  adaptation_patience: 4
```

### Scenario 3: Research and Development

For research scenarios with comprehensive monitoring:

```yaml
federated:
  use_adaptive_strategy: true
  enable_monitoring: true
  target_accuracy: 0.80
  dashboard_update_interval: 15  # More frequent updates
  
trust:
  trust_mode: "hybrid"
  
logging:
  level: DEBUG
  log_to_file: true
  metrics:
    enable_tensorboard: true
```

## Testing the Enhanced Features

### Manual Testing

1. **Start Enhanced Server:**
   ```bash
   python main.py flower_server --config config/enhanced_federated.yaml --verbose
   ```

2. **Connect Test Clients (in separate terminals):**
   ```bash
   python main.py test_client --client-id 1
   python main.py test_client --client-id 2
   python main.py test_client --client-id 3
   ```

3. **Monitor Adaptation:**
   - Watch server logs for threshold adaptation messages
   - Check `trust_dashboard/` directory for generated visualizations
   - Open `trust_dashboard/dashboard.png` to see real-time status

### Automated Testing

```bash
# Run enhanced integration tests
python test_enhanced_integration.py

# Run standard integration tests
python test_integration.py
```

## Understanding Adaptive Behavior

### Trust Threshold Adaptation Logic

The adaptive strategy adjusts thresholds based on:

1. **Below Target Accuracy:** Increases threshold to be more selective
2. **Declining Performance:** Moderately increases threshold  
3. **Good Performance with Positive Trend:** Slightly decreases threshold
4. **Significantly Above Target:** Decreases threshold to include more clients

### Performance Trend Calculation

The system calculates performance trends using:
- Linear regression slope over the performance window
- Positive values indicate improving performance
- Negative values indicate declining performance

### Adaptation Patience

The system waits for `adaptation_patience` rounds between threshold adjustments to:
- Avoid oscillating behavior
- Allow time for threshold changes to take effect
- Maintain system stability

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all dependencies are installed: `pip install matplotlib seaborn`
   - Check Python path includes `src` directory

2. **Dashboard Not Updating:**
   - Verify `enable_monitoring: true` in configuration
   - Check dashboard output directory permissions
   - Ensure matplotlib backend supports file output

3. **Adaptation Not Occurring:**
   - Check if `use_adaptive_strategy: true`
   - Verify sufficient rounds have passed (`adaptation_patience`)
   - Check if performance window has enough data

### Performance Optimization

1. **Dashboard Updates:**
   - Increase `dashboard_update_interval` for better performance
   - Disable monitoring in production for optimal speed

2. **Memory Usage:**
   - Adjust `max_history` in TrustMetricsTracker for memory control
   - Periodically export and clear metrics

## Integration with Existing Features

### Trust Evaluation Modes

All existing trust modes work with adaptive strategies:
- `cosine`: Cosine similarity-based trust
- `entropy`: Entropy-based trust evaluation
- `reputation`: Reputation-based trust scoring
- `hybrid`: Combined trust mechanisms (recommended)

### Flower Compatibility

The enhanced features maintain full compatibility with:
- Standard Flower clients
- Custom Flower strategies
- Flower simulation framework
- Flower dashboard and monitoring tools

## Best Practices

1. **Start Conservative:** Begin with moderate adaptation rates and wider thresholds
2. **Monitor Closely:** Use dashboard during initial deployments
3. **Tune Gradually:** Adjust parameters based on observed behavior
4. **Document Settings:** Keep track of successful parameter combinations
5. **Test Thoroughly:** Use test clients to validate behavior before production

## Advanced Usage

### Custom Adaptation Logic

The `AdaptiveTrustStrategy` can be extended for custom adaptation behavior:

```python
class CustomAdaptiveTrustStrategy(AdaptiveTrustStrategy):
    def _update_trust_threshold(self, round_metrics):
        # Custom adaptation logic here
        super()._update_trust_threshold(round_metrics)
        # Additional custom processing
```

### Custom Monitoring

The `TrustDashboard` can be extended for custom visualizations:

```python
class CustomTrustDashboard(TrustDashboard):
    def generate_custom_plots(self):
        # Custom visualization logic
        pass
```

This enhanced integration provides a powerful foundation for trust-aware federated learning with adaptive behavior and comprehensive monitoring capabilities.
