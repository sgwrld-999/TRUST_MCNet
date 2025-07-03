# TRUST-MCNet Redesigned

A redesigned federated learning framework for **anomaly detection** with **trust mechanisms**, built using modern best practices and clean architecture principles.

## Architecture Overview

This redesigned version of TRUST-MCNet follows SOLID principles and uses:

- **Hydra**: Clean, hierarchical configuration management
- **Ray**: Distributed client execution and resource management  
- **Flower (flwr)**: Modern federated learning simulation framework
- **Trust Mechanisms**: Multi-modal trust evaluation (cosine, entropy, reputation)

## Key Improvements

- ✅ **Clean Architecture**: SOLID principles with clear separation of concerns
- ✅ **Hierarchical Configuration**: Hydra-based config management with composition
- ✅ **Distributed Execution**: Ray actors for scalable client parallelism
- ✅ **Edge Case Handling**: Robust validation and error handling
- ✅ **Production Ready**: Comprehensive logging, monitoring, and error recovery
- ✅ **Minimal Dependencies**: Only essential packages for maintainability

## Directory Structure

```
TRUST_MCNet_Redesigned/
├── config/                     # Hydra configuration hierarchy
│   ├── config.yaml            # Main config with defaults
│   ├── dataset/               # Dataset configurations
│   │   ├── mnist.yaml
│   │   └── custom_csv.yaml
│   ├── env/                   # Environment configurations  
│   │   ├── local.yaml
│   │   ├── iot.yaml
│   │   └── gpu.yaml
│   ├── strategy/              # FL strategy configurations
│   │   ├── fedavg.yaml
│   │   ├── fedadam.yaml
│   │   └── fedprox.yaml
│   ├── trust/                 # Trust mechanism configurations
│   │   ├── hybrid.yaml
│   │   ├── cosine.yaml
│   │   ├── entropy.yaml
│   │   └── reputation.yaml
│   └── model/                 # Model configurations
│       ├── mlp.yaml
│       └── lstm.yaml
├── train.py                   # Hydra entry point
├── simulation.py              # Ray + Flower orchestration
├── clients/                   # Client implementations
│   └── ray_flwr_client.py    # Ray Actor wrapping Flower client
├── utils/                     # Utility functions
│   └── data_utils.py         # Data loading, splitting, validation
├── trust_module/              # Trust evaluation (unchanged)
│   └── trust_evaluator.py
├── models/                    # Model definitions (unchanged)
│   └── model.py
├── data/                      # Data directory (CSV files)
├── requirements.txt           # Minimal dependencies
└── README.md                  # This file
```

## Installation

1. **Clone the repository** and navigate to the redesigned directory:
   ```bash
   cd TRUST_MCNet_Redesigned
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**:
   - MNIST: Will be downloaded automatically
   - CSV data: Place your CSV files in the `data/` directory

## Quick Start

### Basic Training

```bash
# Train with default configuration (MNIST, MLP, FedAvg, Hybrid Trust)
python train.py

# Override specific configurations
python train.py dataset=custom_csv model=lstm strategy=fedadam trust=cosine

# Use GPU environment
python train.py env=gpu

# IoT environment with resource constraints
python train.py env=iot
```

### Configuration Examples

```bash
# Different datasets
python train.py dataset=mnist                    # MNIST with binary classification
python train.py dataset=custom_csv               # Custom CSV dataset

# Different models  
python train.py model=mlp                        # Multi-layer perceptron
python train.py model=lstm                       # LSTM for sequential data

# Different strategies
python train.py strategy=fedavg                  # Federated Averaging
python train.py strategy=fedadam                 # FedAdam optimization
python train.py strategy=fedprox                 # FedProx with proximal term

# Different trust mechanisms
python train.py trust=hybrid                     # Hybrid trust (default)
python train.py trust=cosine                     # Cosine similarity only
python train.py trust=entropy                    # Entropy-based only
python train.py trust=reputation                 # Reputation-based only
```

### Hyperparameter Sweeps

```bash
# Multi-run sweep across strategies and trust mechanisms
python train.py -m strategy=fedavg,fedadam trust=cosine,entropy,hybrid

# Grid search over learning rates and client counts
python train.py -m training.learning_rate=0.001,0.01,0.1 dataset.num_clients=5,10,20
```

## Configuration System

The configuration system uses Hydra's composition pattern:

### Main Configuration (`config/config.yaml`)
```yaml
defaults:
  - dataset: mnist      # Default dataset
  - env: local         # Default environment  
  - strategy: fedavg   # Default FL strategy
  - trust: hybrid      # Default trust mechanism
  - model: mlp         # Default model
  - _self_            # Include base config

# Global settings
training:
  epochs: 1
  learning_rate: 0.001
  
federated:
  num_rounds: 3
  fraction_fit: 0.8
```

### Dataset Configuration (`config/dataset/mnist.yaml`)
```yaml
dataset:
  name: mnist
  path: "./data/MNIST"
  num_clients: 5
  eval_fraction: 0.2
  batch_size: 32
  partitioning: iid      # or dirichlet, pathological
```

### Environment Configuration (`config/env/local.yaml`)
```yaml
env:
  name: local
  device: auto           # auto, cpu, cuda, mps
  ray:
    num_cpus: 4
    num_gpus: 0
    object_store_memory: 1000000000
```

## Architecture Principles

### 1. Single Responsibility Principle (SRP)
- `train.py`: Entry point and configuration management
- `simulation.py`: Orchestration of Ray + Flower
- `ray_flwr_client.py`: Client-side training and evaluation
- `data_utils.py`: Data loading and preprocessing
- `trust_evaluator.py`: Trust mechanism implementation

### 2. Open/Closed Principle (OCP)
- Easy to add new datasets via configuration
- New trust mechanisms through config composition
- New FL strategies without code changes

### 3. Dependency Inversion Principle (DIP)
- Configuration-driven dependency injection
- Abstract interfaces for extensibility
- Minimal coupling between components

### 4. Edge Case Handling
- Validates `num_clients >= 1` and sufficient data per client
- Handles empty datasets and invalid splits gracefully
- Robust error recovery and logging

## Key Features

### 1. **Distributed Client Execution**
- Ray actors for true parallelism
- Resource management and isolation
- Scalable to hundreds of clients

### 2. **Trust-Aware Federated Learning**
- Multiple trust evaluation modes
- Client selection based on trust scores
- Malicious client detection

### 3. **Flexible Configuration**
- Hierarchical config composition
- Easy hyperparameter sweeps
- Environment-specific settings

### 4. **Production-Ready**
- Comprehensive logging and monitoring
- Error handling and recovery
- Resource cleanup and management

## Extending the Framework

### Adding a New Dataset
1. Create `config/dataset/my_dataset.yaml`
2. Implement loading logic in `data_utils.py`
3. Use: `python train.py dataset=my_dataset`

### Adding a New Strategy
1. Create `config/strategy/my_strategy.yaml`
2. Add strategy creation in `simulation.py`
3. Use: `python train.py strategy=my_strategy`

### Adding a New Trust Mechanism
1. Create `config/trust/my_trust.yaml`
2. Extend `trust_evaluator.py` if needed
3. Use: `python train.py trust=my_trust`

## Monitoring and Debugging

### Logs
- Console output with configurable levels
- File logging to `simulation.log`
- Ray dashboard at `http://localhost:8265`

### Hydra Output Management
- Automatic experiment tracking
- Working directory per run
- Configuration preservation

### Performance Monitoring
- Client-level resource usage
- Training time metrics
- Trust score evolution

## License

Same as original TRUST-MCNet project.

## Contributing

1. Follow SOLID principles
2. Add comprehensive tests
3. Update configuration documentation
4. Ensure backward compatibility
