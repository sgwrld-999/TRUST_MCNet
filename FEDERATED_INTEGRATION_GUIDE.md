# TRUST_MCNet Federated Learning Integration Guide

This guide shows how to use the integrated TRUST_MCNet system with Flower federated learning support.

## Overview

The main.py now supports multiple operational modes:
- `simulation`: Run the standard TRUST_MCNet simulation
- `flower_server`: Run trust-weighted Flower federated learning server
- `test_client`: Run test client for Flower server validation

## Usage Examples

### 1. Standard TRUST_MCNet Simulation

```bash
# Run with default settings
python main.py simulation

# Run with custom parameters
python main.py simulation --clients 10 --rounds 10 --verbose

# Show simulation help
python main.py simulation --help
```

### 2. Trust-Weighted Flower Server

```bash
# Run Flower server with default settings
python main.py flower_server

# Run with custom configuration
python main.py flower_server --num-rounds 10 --address 0.0.0.0:8080 --verbose

# Show server help
python main.py flower_server --help
```

### 3. Test Client for Flower Server

```bash
# Run test client (requires server to be running)
python main.py test_client --client-id 1

# Connect to specific server
python main.py test_client --client-id 2 --verbose
```

## Configuration

The system uses the enhanced `config/config.yaml` file which now includes:

### Federated Learning Settings
```yaml
federated:
  num_rounds: 5
  fraction_fit: 0.8
  fraction_evaluate: 0.2
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2
  server:
    address: "0.0.0.0:8080"
    accept_failures: true
```

### Client Simulation Settings
```yaml
simulation:
  client_simulation:
    server_address: "localhost:8080"
    client_bias_factor: 0.05
```

## Trust Mechanisms

The trust-weighted aggregation uses the existing TrustEvaluator with configurable trust modes:

```yaml
trust:
  trust_mode: "hybrid"  # cosine, entropy, reputation, hybrid
  threshold: 0.5
  learning_rate: 0.01
  use_dynamic_weights: true
```

## Testing the Integration

### Step 1: Start the Trust-Weighted Server
```bash
python main.py flower_server --verbose
```

### Step 2: Connect Test Clients (in separate terminals)
```bash
# Terminal 2
python main.py test_client --client-id 1

# Terminal 3  
python main.py test_client --client-id 2

# Terminal 4
python main.py test_client --client-id 3
```

### Step 3: Monitor Trust-Based Aggregation
The server will show trust evaluation logs and aggregation results based on client performance and trust scores.

## Key Features

1. **Trust-Aware Aggregation**: Uses TRUST_MCNet's trust evaluation mechanisms
2. **SOLID Design**: Clean interface-based architecture
3. **Robust Error Handling**: Fallback mechanisms for import and runtime errors
4. **Configuration-Driven**: Extensive YAML configuration support
5. **Production Ready**: Comprehensive logging and monitoring

## Dependencies

- `flwr`: Flower federated learning framework
- `torch`: PyTorch for neural network operations
- `numpy`: Numerical computations
- `scipy`: Statistical functions for trust evaluation
- `yaml`: Configuration file parsing

Install missing dependencies:
```bash
pip install flwr torch numpy scipy pyyaml
```

## Architecture Integration

The integration follows the SOLID principles:

- **Single Responsibility**: TrustWeightedStrategy only handles trust-aware aggregation
- **Open/Closed**: Extends Flower's FedAvg without modifying core logic
- **Liskov Substitution**: Can replace any Flower strategy
- **Interface Segregation**: Minimal interface coupling
- **Dependency Inversion**: Depends on TrustEvaluator abstraction

## File Structure After Integration

```
TRUST_MCNet/
├── main.py                          # Enhanced main entry point
├── config/
│   └── config.yaml                  # Enhanced configuration
├── src/trust_mcnet/
│   ├── strategies/
│   │   └── trust_weighted_strategy.py  # Flower integration
│   └── trust_module/
│       └── trust_evaluator.py      # Trust mechanisms
└── examples/
    └── start_simulation.py         # Original simulation
```

## Troubleshooting

### Import Errors
The system includes fallback import mechanisms. If you encounter import errors, ensure all dependencies are installed.

### Connection Issues
- Check server address and port
- Ensure firewall allows connections
- Verify server is running before starting clients

### Trust Evaluation Issues
- Check trust mode configuration
- Verify threshold values are appropriate
- Monitor trust scores in verbose logs
