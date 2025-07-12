# FLTrust Integration for TRUST-MCNet

This document describes the integration of FLTrust (Byzantine-robust Federated Learning via Trust Bootstrapping) aggregation mechanism into the TRUST-MCNet federated learning framework.

## Overview

FLTrust is a robust federated learning aggregation algorithm that uses a trusted root dataset to evaluate client trustworthiness and perform Byzantine-resistant model aggregation. This implementation follows the original paper specifications: [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/pdf/2012.13995).

## Key Features

### ✅ Core FLTrust Algorithm
- **Root dataset computation**: Computes trusted reference gradients Δw_root
- **Cosine similarity trust scoring**: s_k = cos_sim(Δw_k, Δw_root)
- **Trust score clipping**: Clips scores to non-negative values
- **Magnitude normalization**: Δw_k ← (‖Δw_root‖₂ / ‖Δw_k‖₂) · Δw_k
- **Trust-weighted aggregation**: Δw_global = Σ_k (s_k / Σ_j s_j) · Δw_k
- **Server update**: w_{t+1} = w_t + η · Δw_global

### ✅ TRUST-MCNet Integration
- **Modular design**: Pluggable aggregator interface for easy switching
- **Trust score override**: Integration with TRUST-MCNet's trust evaluation module
- **Gradient clipping**: Configurable maximum norm clipping
- **Warm-up rounds**: Stable trust score development
- **Device management**: GPU/CPU compatibility

## Quick Start

### Basic Usage

```python
from TRUST_MCNet_Codebase.server.fltrust_server import create_fltrust_server
from TRUST_MCNet_Codebase.utils.aggregator import FLTrustAggregator
import torch
from torch.utils.data import DataLoader

# Create your model and root dataset
model = YourModel()
root_dataset_loader = DataLoader(trusted_dataset, batch_size=32)

# Configuration
config = {
    'server_learning_rate': 1.0,
    'clip_threshold': 10.0,
    'warm_up_rounds': 5,
    'trust_threshold': 0.1,
    'trust_mode': 'hybrid'
}

# Create FLTrust server
server = create_fltrust_server(
    global_model=model,
    root_dataset_loader=root_dataset_loader,
    config=config
)

# Run federated training
for round_num in range(num_rounds):
    results = server.run_federated_round(available_clients, client_trainer_fn)
    print(f"Round {round_num}: {results}")
```

### Manual Aggregator Setup

```python
from TRUST_MCNet_Codebase.utils.aggregator import FLTrustAggregator, FedAvgAggregator
from TRUST_MCNet_Codebase.server.server import FederatedServer

# Create server
server = FederatedServer(model, config)

# Create FLTrust aggregator
fltrust_aggregator = FLTrustAggregator(
    root_model=root_model,
    root_dataset_loader=root_dataset_loader,
    server_learning_rate=1.0,
    clip_threshold=10.0,
    warm_up_rounds=5,
    trust_threshold=0.0
)

# Set aggregator
server.aggregator = fltrust_aggregator

# Switch to FedAvg if needed
fedavg_aggregator = FedAvgAggregator(clip_threshold=10.0)
server.aggregator = fedavg_aggregator
```

## Configuration Options

### FLTrust-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_learning_rate` | float | 1.0 | Global learning rate (η) for server updates |
| `clip_threshold` | float | 10.0 | Maximum norm for gradient clipping |
| `warm_up_rounds` | int | 5 | Rounds before trust scoring takes effect |
| `trust_threshold` | float | 0.0 | Minimum trust score threshold |

### TRUST-MCNet Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trust_mode` | str | 'hybrid' | TRUST-MCNet trust evaluation mode |
| `trust_threshold` | float | 0.5 | TRUST-MCNet trust threshold |
| `use_trust_override` | bool | True | Use TRUST-MCNet scores instead of cosine similarity |

### Root Dataset Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dataset_size` | int | 100 | Size of trusted root dataset |
| `root_batch_size` | int | 32 | Batch size for root dataset processing |

## Advanced Usage

### Custom Trust Score Integration

```python
# Combine FLTrust cosine similarity with TRUST-MCNet trust scores
def combine_trust_scores(cosine_scores, trust_mcnet_scores, alpha=0.7):
    combined_scores = {}
    for client_id in cosine_scores:
        if client_id in trust_mcnet_scores:
            combined = alpha * trust_mcnet_scores[client_id] + (1 - alpha) * cosine_scores[client_id]
            combined_scores[client_id] = combined
        else:
            combined_scores[client_id] = cosine_scores[client_id]
    return combined_scores

# Use in aggregation
custom_scores = combine_trust_scores(cosine_scores, trust_mcnet_scores)
result = fltrust_aggregator.aggregate(
    client_updates=client_updates,
    global_weights=global_weights,
    trust_scores_override=custom_scores
)
```

### Scenario-Based Configurations

```python
from TRUST_MCNet_Codebase.config.fltrust_config import SCENARIO_CONFIGS

# High security scenario
config = SCENARIO_CONFIGS['high_security']
server = create_fltrust_server(model, root_dataset, config)

# Fast convergence scenario
config = SCENARIO_CONFIGS['fast_convergence']
server = create_fltrust_server(model, root_dataset, config)
```

### Aggregator Switching

```python
# Switch between different aggregators during training
aggregators = {
    'fltrust': FLTrustAggregator(...),
    'fedavg': FedAvgAggregator(...),
    'trimmed_mean': TrimmedMeanAggregator(...)
}

for round_num in range(num_rounds):
    # Use FLTrust for first 10 rounds, then switch to FedAvg
    if round_num < 10:
        server.set_aggregator(aggregators['fltrust'])
    else:
        server.set_aggregator(aggregators['fedavg'])
    
    results = server.run_federated_round(available_clients, client_trainer_fn)
```

## Testing

Run the comprehensive test suite:

```bash
python TRUST_MCNet_Codebase/tests/test_fltrust.py
```

Run the integration example:

```bash
python TRUST_MCNet_Codebase/examples/fltrust_integration_example.py
```

## Architecture

### Class Hierarchy

```
AggregatorInterface (ABC)
├── FLTrustAggregator
├── FedAvgAggregator
├── TrimmedMeanAggregator
└── KrumAggregator

FederatedServer
└── FLTrustFederatedServer
```

### Key Components

1. **FLTrustAggregator**: Core FLTrust algorithm implementation
2. **FLTrustFederatedServer**: Enhanced server with FLTrust update mechanism
3. **AggregatorInterface**: Base interface for all aggregators
4. **Configuration**: Flexible configuration system for different scenarios

## Algorithm Details

### FLTrust Algorithm Flow

1. **Initialization**: Set up root model and dataset
2. **Root Update Computation**: Compute Δw_root from trusted dataset
3. **Client Update Processing**: 
   - Compute Δw_k = w_k - w_t for each client
   - Apply gradient clipping
   - Normalize magnitude to match root update
4. **Trust Score Computation**: s_k = cos_sim(Δw_k, Δw_root)
5. **Trust Score Processing**:
   - Clip to non-negative values
   - Apply trust threshold
   - Normalize to sum to 1
6. **Aggregation**: Δw_global = Σ_k (s_k / Σ_j s_j) · Δw_k
7. **Server Update**: w_{t+1} = w_t + η · Δw_global

### Trust Score Integration

The implementation supports multiple trust score sources:

1. **Cosine Similarity Only**: Pure FLTrust algorithm
2. **TRUST-MCNet Only**: Use existing trust evaluation
3. **Hybrid Combination**: Weighted combination of both
4. **Fallback Strategy**: Use TRUST-MCNet as primary, cosine as fallback

## Performance Considerations

### Computational Complexity

- **Root update computation**: O(|D_root| × |θ|) per round
- **Cosine similarity**: O(|θ|) per client
- **Magnitude normalization**: O(|θ|) per client
- **Aggregation**: O(K × |θ|) where K is number of clients

### Memory Usage

- Root model: Same size as global model
- Client updates: K × |θ| parameters
- Trust scores: K scalars

### Optimization Tips

1. **Root dataset size**: Balance between accuracy and efficiency (50-500 samples)
2. **Batch processing**: Use appropriate batch sizes for root dataset
3. **Gradient accumulation**: Accumulate gradients over multiple batches if needed
4. **Device management**: Ensure all tensors are on the same device

## Troubleshooting

### Common Issues

1. **Zero root update norm**: Increase root dataset size or learning rate
2. **All zero trust scores**: Check trust threshold and client update magnitudes
3. **Memory issues**: Reduce root dataset size or use gradient accumulation
4. **Device mismatches**: Ensure root model and data are on the same device

### Debug Tips

```python
# Enable detailed logging
import logging
logging.getLogger('TRUST_MCNet_Codebase.utils.aggregator').setLevel(logging.DEBUG)

# Check aggregation info
info = fltrust_aggregator.get_aggregation_info()
print(f"Aggregator state: {info}")

# Monitor trust scores
trust_scores = fltrust_aggregator.compute_cosine_trust_scores(client_updates, root_update)
print(f"Trust scores: {trust_scores}")
```

## References

1. [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/pdf/2012.13995)
2. [TRUST-MCNet Original Framework]
3. [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)

## Contributing

To extend the FLTrust implementation:

1. Inherit from `AggregatorInterface` for new aggregators
2. Follow the established parameter naming conventions
3. Add comprehensive tests for new features
4. Update configuration files for new parameters
5. Document any breaking changes

## License

This implementation follows the same license as the TRUST-MCNet framework.
