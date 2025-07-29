# Dynamic Trust Threshold Mechanism

## Overview

The dynamic trust threshold mechanism in TRUST_MCNet provides an adaptive approach to determining client trustworthiness in federated learning environments. Rather than using a fixed threshold value, this mechanism adjusts the threshold based on the distribution of trust scores, the current federated learning round, and global model performance.

## Key Features

- **Adaptive to Trust Score Distribution**: Adjusts threshold based on the current distribution of trust scores
- **Round-Aware Behavior**: Different behavior in early, middle, and later rounds of federated learning
- **Minimum Client Guarantees**: Ensures at least a minimum number of trusted clients
- **Performance-Sensitive**: Can incorporate global model performance data
- **Configurable Parameters**: Easily customizable through API or configuration

## Implementation Details

The dynamic threshold mechanism combines three complementary approaches:

1. **Percentile-Based Threshold**: Uses statistical percentiles of the trust score distribution
2. **Statistical Threshold**: Based on mean and standard deviation of trust scores
3. **Adaptive Threshold**: Adjusts based on historical patterns and performance trends

These three methods are combined with weights that vary depending on the federated learning round:

```python
if round_number <= 3:
    # Early rounds: be more lenient, focus on percentile and statistical
    dynamic_threshold = (
        0.5 * threshold_percentile + 
        0.4 * threshold_statistical + 
        0.1 * threshold_adaptive
    )
elif round_number <= 10:
    # Middle rounds: balanced approach
    dynamic_threshold = (
        percentile_weight * threshold_percentile + 
        statistical_weight * threshold_statistical + 
        adaptive_weight * threshold_adaptive
    )
else:
    # Later rounds: more sophisticated, history-based
    dynamic_threshold = (
        0.2 * threshold_percentile + 
        0.3 * threshold_statistical + 
        0.5 * threshold_adaptive
    )
```

## Configuration Parameters

The dynamic threshold mechanism can be configured with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_trusted_ratio` | Target ratio of clients to consider trusted | 0.6 |
| `min_trusted_clients` | Minimum number of trusted clients to guarantee | 2 |
| `min_threshold` | Minimum allowable threshold value | 0.1 |
| `max_threshold` | Maximum allowable threshold value | 0.9 |
| `percentile_weight` | Weight for percentile-based threshold | 0.4 |
| `statistical_weight` | Weight for statistical threshold | 0.4 |
| `adaptive_weight` | Weight for adaptive threshold | 0.2 |

## Using Dynamic Thresholds

### Via Configuration File

```yaml
trust:
  mode: dynamic
  dynamic_threshold:
    enabled: true
    target_trusted_ratio: 0.6
    min_trusted_clients: 2
    min_trust_threshold: 0.1
    max_trust_threshold: 0.9
    percentile_weight: 0.4
    statistical_weight: 0.4
    adaptive_weight: 0.2
```

### Via API

The enhanced API server provides endpoints for configuring the dynamic threshold mechanism:

```
GET /threshold           # Get current threshold configuration
POST /threshold/dynamic  # Update dynamic threshold configuration
```

Example configuration update:

```json
{
  "config": {
    "target_trusted_ratio": 0.7,
    "min_trusted_clients": 3,
    "min_threshold": 0.15,
    "max_threshold": 0.85,
    "percentile_weight": 0.3,
    "statistical_weight": 0.5,
    "adaptive_weight": 0.2
  },
  "enable_dynamic_threshold": true,
  "reason": "Adjusting for increased client participation"
}
```

## Benefits

1. **Robustness**: Adapts to varying trust score distributions automatically
2. **Fairness**: Ensures a minimum number of clients are included in aggregation
3. **Security**: Becomes stricter in later rounds when model is more mature
4. **Visibility**: Provides clear insights through logging and API endpoints
5. **Configurability**: Can be fine-tuned for different deployment scenarios

## Monitoring and Analysis

The enhanced API provides analytical tools to evaluate the impact of the dynamic threshold on system performance:

```
GET /analysis/threshold?rounds=10  # Analyze impact of threshold on performance
```

This endpoint returns correlation data between threshold changes and system performance metrics, helping administrators fine-tune the dynamic threshold parameters for optimal results.
