# TRUST-MCNet with Flwr Integration

This document describes the Flwr (Flower) federated learning integration for TRUST-MCNet, specifically designed for IoT devices with resource constraints and trust mechanisms.

## Overview

The integration includes:
- **FedAdam Strategy**: Advanced federated optimization with adaptive learning rates
- **IoT Device Support**: Resource monitoring and adaptive training for constrained devices
- **Trust Mechanisms**: Client trust evaluation and selection based on performance
- **Simulation Framework**: Complete simulation environment for federated learning research

## Key Features

### 1. Flwr Server with FedAdam Strategy
- Custom `TrustAwareFedAdam` strategy extending Flwr's FedAdam
- Trust-based client selection and weighting
- Resource-aware aggregation
- Configurable parameters: `eta`, `eta_l`, `beta_1`, `beta_2`, `tau`

### 2. IoT-Optimized Clients
- Real-time resource monitoring (CPU, memory)
- Adaptive batch size based on resource constraints
- Efficient training with memory cleanup
- Performance tracking and reporting

### 3. Trust Evaluation
- Performance-based trust scoring
- Resource efficiency evaluation
- Training time considerations
- Historical trust tracking

### 4. Configuration Parameters

#### Federated Learning Settings
```yaml
federated:
  num_clients: 10
  min_available_clients: 5
  strategy: "FedAdam"
  strategy_config:
    eta: 0.001      # Server-side learning rate
    eta_l: 0.001    # Client-side learning rate
    beta_1: 0.9     # Momentum parameter
    beta_2: 0.999   # Second moment parameter
    tau: 0.001      # Control variates parameter
```

#### IoT Device Configuration
```yaml
iot_config:
  max_memory_mb: 512        # Maximum memory usage per client
  max_cpu_percent: 70       # Maximum CPU usage per client
  adaptive_batch_size: true # Enable adaptive batch sizing
  min_batch_size: 8         # Minimum batch size
  max_batch_size: 64        # Maximum batch size
```

## Installation

### Quick Setup
```bash
# Make setup script executable and run
chmod +x setup_flwr.sh
./setup_flwr.sh

# Or use Makefile
make install-flwr
```

### Manual Installation
```bash
# Core dependencies
pip install torch>=1.9.0 torchvision>=0.10.0
pip install flwr>=1.6.0 flwr-datasets>=0.0.2
pip install psutil>=5.9.0 "ray[default]">=2.8.0

# Install all requirements
pip install -r requirements.txt
```

## Usage

### Basic Simulation
```bash
# Run with default configuration (10 clients, 100 rounds)
make run-flwr

# Or directly
cd TRUST_MCNet_Codebase
python scripts/flwr_simulation.py --config config/config.yaml
```

### Custom Parameters
```bash
# Custom number of clients and rounds
python scripts/flwr_simulation.py --config config/config.yaml --clients 20 --rounds 50

# Non-IID data distribution
python scripts/flwr_simulation.py --config config/config.yaml --data-distribution non_iid

# Quick test with fewer resources
make run-flwr-custom
```

### Configuration Options
```bash
--config PATH           # Configuration file path
--clients N            # Number of clients (overrides config)
--rounds N             # Number of rounds (overrides config)
--data-distribution    # "iid" or "non_iid"
```

## Architecture Components

### 1. Server (`server/flwr_server.py`)
- `TrustAwareFedAdam`: Custom strategy with trust mechanisms
- Client selection based on trust scores
- Resource-aware model aggregation
- Performance monitoring and logging

### 2. Client (`clients/flwr_client.py`)
- `TrustMCNetFlwrClient`: IoT-optimized Flwr client
- `IoTResourceMonitor`: Real-time resource monitoring
- Adaptive training parameters
- Trust-aware performance reporting

### 3. Simulation (`scripts/flwr_simulation.py`)
- Complete simulation framework
- Synthetic IoT dataset generation
- Client dataset distribution (IID/non-IID)
- Results logging and analysis

### 4. Enhanced Client (`clients/client.py`)
- Updated traditional client with IoT optimizations
- Resource monitoring integration
- Adaptive batch sizing
- Attack simulation capabilities

## Trust Mechanisms

### Trust Score Calculation
Trust scores are calculated based on:
- **Performance**: Model accuracy on local data
- **Resource Efficiency**: CPU and memory usage
- **Training Time**: Faster training indicates better resource management

### Client Selection
- First round: Random selection
- Subsequent rounds: Trust-based selection
- Minimum client requirements enforced
- Adaptive resource consideration

### Trust-Weighted Aggregation
- Model updates weighted by trust scores
- Failed clients receive low trust scores
- Historical trust tracking (last 10 rounds)

## IoT Device Optimizations

### Resource Monitoring
- Continuous CPU and memory monitoring
- Background thread for non-intrusive monitoring
- Resource constraint detection

### Adaptive Training
- Dynamic batch size adjustment
- Epoch reduction under resource constraints
- Memory cleanup between batches
- CPU throttling for high usage

### Efficiency Measures
- Single-threaded data loading (`num_workers=0`)
- Minimal memory footprint
- CUDA memory cleanup
- Optimized tensor operations

## Results and Metrics

### Training Metrics
- Accuracy and loss per client
- Training time and resource usage
- Adaptive batch sizes used
- Trust scores evolution

### Centralized Metrics
- Global model performance
- Client participation rates
- Trust distribution statistics
- Resource utilization summaries

### Output Files
- `simulation_results.txt`: Comprehensive results summary
- Training logs: Real-time progress monitoring
- TensorBoard logs: Detailed metrics visualization

## Example Configuration

```yaml
# Complete example configuration
model:
  type: "MLP"
  mlp:
    input_dim: 10
    hidden_dims: [64, 32, 16]
    output_dim: 2

federated:
  num_clients: 10
  num_rounds: 100
  min_available_clients: 5
  strategy: "FedAdam"
  strategy_config:
    eta: 0.001
    eta_l: 0.001
    beta_1: 0.9
    beta_2: 0.999
    tau: 0.001
  iot_config:
    max_memory_mb: 512
    max_cpu_percent: 70
    adaptive_batch_size: true
    min_batch_size: 8
    max_batch_size: 64

client:
  learning_rate: 0.001
  batch_size: 32
  local_epochs: 5

data:
  dataset_size: 5000
  anomaly_ratio: 0.1
  distribution: "iid"

trust:
  enabled: true
  trust_threshold: 0.7
```

## Performance Considerations

### For IoT Devices
- Memory usage typically < 512MB
- CPU usage typically < 70%
- Training time optimized for quick iterations
- Network communication minimized

### Scalability
- Supports 5-50 clients efficiently
- Adaptive to heterogeneous device capabilities
- Graceful degradation under resource constraints

## Troubleshooting

### Common Issues
1. **Flwr Import Error**: Ensure Flwr is installed (`pip install flwr`)
2. **Resource Constraints**: Reduce batch size or number of clients
3. **Memory Issues**: Enable adaptive batch sizing
4. **Slow Training**: Check resource monitoring logs

### Verification
```bash
# Test installation
make test-flwr

# Check versions
python -c "import flwr; print(flwr.__version__)"
python -c "import torch; print(torch.__version__)"
```

## Research Applications

This implementation is suitable for:
- IoT anomaly detection research
- Federated learning with trust mechanisms
- Resource-constrained federated learning
- Non-IID data distribution studies
- Federated optimization algorithm comparison

## Future Enhancements

- Support for other FL strategies (FedProx, FedNova)
- Advanced attack simulation
- Real IoT device deployment
- Energy consumption modeling
- Adaptive communication protocols
