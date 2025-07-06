# TRUST-MCNet Simulation Guide

A comprehensive guide to running federated learning simulations with trust mechanisms and IoT device optimization.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Project Structure](#project-structure)
5. [Running Simulations](#running-simulations)
6. [Configuration Options](#configuration-options)
7. [Available Datasets](#available-datasets)
8. [Model Architectures](#model-architectures)
9. [Trust Mechanisms](#trust-mechanisms)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)

## Overview

TRUST-MCNet is an advanced federated learning framework designed for anomaly detection with trust mechanisms. It supports:
- **Federated Learning**: Distributed training across multiple IoT clients
- **Trust Mechanisms**: Multi-modal trust evaluation (cosine similarity, entropy, reputation)
- **IoT Optimization**: Resource monitoring, adaptive batch sizing, memory management
- **Multiple Architectures**: Two implementations (Original and Redesigned)

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM (16GB recommended for large simulations)
- **Storage**: 5GB+ free space
- **GPU**: Optional, CUDA-compatible for acceleration

### Software Requirements
- **Python**: 3.8+ (Python 3.9+ recommended)
- **Operating System**: macOS, Linux, or Windows
- **Package Manager**: pip or conda

## Installation & Setup

### Quick Setup (Recommended)

```bash
# Clone the repository
cd /path/to/TRUST_MCNet

# Run the setup script (for Flower integration)
chmod +x setup_flwr.sh
./setup_flwr.sh

# Verify installation
python3 -c "import flwr; print(f'Flwr version: {flwr.__version__}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Manual Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For the redesigned version
cd TRUST_MCNet_Redesigned
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest black flake8 mypy
```

### Verify GPU Support (Optional)

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

The project contains two main implementations:

### 1. Original Implementation (`TRUST_MCNet_Codebase/`)
- **Entry Point**: `scripts/main.py` or `scripts/flwr_simulation.py`
- **Configuration**: `config/config.yaml`
- **Features**: Basic federated learning with trust mechanisms

### 2. Redesigned Implementation (`TRUST_MCNet_Redesigned/`)
- **Entry Point**: `train.py` (Hydra-based) or `enhanced_simulation.py`
- **Configuration**: Hierarchical config system (`config/`)
- **Features**: Production-grade, SOLID principles, enhanced architecture

## Running Simulations

### Method 1: Original Implementation

#### Basic Simulation
```bash
cd TRUST_MCNet_Codebase
python scripts/flwr_simulation.py --config config/config.yaml
```

#### Main Entry Point
```bash
cd TRUST_MCNet_Codebase
python scripts/main.py
```

### Method 2: Redesigned Implementation (Recommended)

#### Quick Start - Enhanced Simulation
```bash
cd TRUST_MCNet_Redesigned

# Run with default settings (MNIST, 5 clients, 3 rounds)
python enhanced_simulation.py

# Alternative: Using Hydra entry point
python train.py
```

#### Multi-Dataset Simulation
```bash
# Run across multiple datasets
python run_multi_dataset_simulation.py --clients 5 --rounds 3 --datasets mnist custom_csv

# With specific parameters
python run_multi_dataset_simulation.py --clients 10 --rounds 5 --datasets mnist iot_general
```

#### Legacy Compatibility
```bash
# Use legacy simulation mode
python train.py simulation.use_legacy=true
```

### Method 3: Configuration-Based Runs

#### Using Hydra Configuration Groups
```bash
cd TRUST_MCNet_Redesigned

# Basic configurations
python train.py dataset=mnist model=mlp strategy=fedavg
python train.py dataset=custom_csv model=lstm strategy=fedadam

# Environment-specific runs
python train.py env=gpu training.local_epochs=5
python train.py env=iot dataset.num_clients=20

# Trust mechanism selection
python train.py trust=cosine
python train.py trust=entropy
python train.py trust=hybrid

# Combined configurations
python train.py dataset=mnist model=lstm strategy=fedadam trust=hybrid env=gpu
```

#### Multi-Epoch Training
```bash
# Enable multi-epoch local training
python train.py training=multi_epoch training.local_epochs=3

# With specific learning rate
python train.py training=multi_epoch training.learning_rate=0.001
```

### Method 4: Examples and Demos

#### Run Example Configurations
```bash
cd TRUST_MCNet_Redesigned

# View available examples
python examples.py

# Run demo with refactored architecture
python demo_refactored.py

# Run architecture validation
python validate_architecture.py
```

## Configuration Options

### Dataset Configuration (`config/dataset/`)

#### MNIST Dataset (`mnist.yaml`)
```yaml
name: mnist
path: ./data
num_clients: 5
eval_fraction: 0.2
batch_size: 32
partitioning: iid  # or "dirichlet", "pathological"
alpha: 0.5  # For Dirichlet partitioning
transforms:
  normalize: true
  augment: false
```

#### Custom CSV Dataset (`custom_csv.yaml`)
```yaml
name: custom_csv
path: ./data/your_dataset.csv
target_column: label
num_clients: 10
eval_fraction: 0.2
batch_size: 64
partitioning: dirichlet
alpha: 0.3
preprocessing:
  normalize: true
  scale: true
```

### Model Configuration (`config/model/`)

#### MLP Model (`mlp.yaml`)
```yaml
type: MLP
mlp:
  input_dim: 784
  hidden_dims: [128, 64, 32]
  output_dim: 10
  dropout_rate: 0.2
  activation: relu
```

#### LSTM Model (`lstm.yaml`)
```yaml
type: LSTM
lstm:
  input_dim: 10
  hidden_dim: 64
  num_layers: 2
  output_dim: 1
  dropout_rate: 0.1
  bidirectional: false
```

### Federated Learning Strategy (`config/strategy/`)

#### FedAvg (`fedavg.yaml`)
```yaml
name: fedavg
fraction_fit: 0.8
fraction_evaluate: 0.2
min_fit_clients: 2
min_evaluate_clients: 2
min_available_clients: 2
```

#### FedAdam (`fedadam.yaml`)
```yaml
name: fedadam
fraction_fit: 0.8
fraction_evaluate: 0.2
min_fit_clients: 3
min_evaluate_clients: 3
min_available_clients: 3
eta: 1e-2
eta_l: 1e-1
beta_1: 0.9
beta_2: 0.99
tau: 1e-9
```

### Trust Mechanisms (`config/trust/`)

#### Hybrid Trust (`hybrid.yaml`)
```yaml
mode: hybrid
threshold: 0.5
weights:
  cosine: 0.4
  entropy: 0.3
  reputation: 0.3
enable_filtering: true
```

### Environment Configuration (`config/env/`)

#### GPU Environment (`gpu.yaml`)
```yaml
device: cuda
ray:
  num_cpus: 4
  num_gpus: 1
  object_store_memory: 1000000000
dataloader:
  num_workers: 4
  pin_memory: true
training:
  enable_gpu_optimization: true
```

#### IoT Environment (`iot.yaml`)
```yaml
device: cpu
ray:
  num_cpus: 2
  num_gpus: 0
  object_store_memory: 500000000
dataloader:
  num_workers: 0
  pin_memory: false
resource_monitoring:
  enabled: true
  memory_limit: 512MB
  adaptive_batch_size: true
```

## Available Datasets

### 1. MNIST (Built-in)
- **Type**: Image classification → Binary anomaly detection
- **Size**: 60,000 training + 10,000 test samples
- **Features**: 28x28 grayscale images (784 features)
- **Classes**: Converted to binary (digits 1,7 as anomalies)
- **Usage**: Automatic download and preprocessing

### 2. Custom CSV Datasets
- **Location**: Place CSV files in `data/` directory
- **Requirements**: Must have a target column named 'label'
- **Format**: Any tabular data with numerical features
- **Examples**: IoT sensor data, network traffic, medical data

### 3. Synthetic IoT Data
- **Type**: Generated sensor data with anomaly patterns
- **Features**: Configurable input dimensions
- **Anomaly Ratio**: Configurable (default: 10%)
- **Usage**: Automatically generated during simulation

### 4. Supported External Datasets
Based on the data files in `data/Datasets/`:
- **CIC_IoMT_2024.csv**: IoT Medical Things dataset
- **CIC_IoT_2023.csv**: IoT network traffic
- **Edge_IIoT.csv**: Industrial IoT edge data
- **IoT_23.csv**: IoT device data
- **MedBIoT.csv**: Medical IoT botnet data

## Model Architectures

### 1. Multi-Layer Perceptron (MLP)
```python
# Configuration
model:
  type: MLP
  mlp:
    input_dim: 784  # MNIST: 784, adjust for your data
    hidden_dims: [128, 64, 32]
    output_dim: 2   # Binary classification
    dropout_rate: 0.2
    activation: relu
```

**Best for**: Tabular data, simple feature patterns, fast training

### 2. Long Short-Term Memory (LSTM)
```python
# Configuration
model:
  type: LSTM
  lstm:
    input_dim: 10   # Number of features
    hidden_dim: 64
    num_layers: 2
    output_dim: 1   # Binary output
    dropout_rate: 0.1
    bidirectional: false
```

**Best for**: Sequential data, time series, temporal patterns

## Trust Mechanisms

### 1. Cosine Similarity Trust
- **Principle**: Measures similarity between client updates and global model
- **Range**: [0, 1] (higher = more trustworthy)
- **Best for**: Detecting malicious parameter updates

```bash
python train.py trust=cosine
```

### 2. Entropy-Based Trust
- **Principle**: Higher parameter entropy indicates better model diversity
- **Range**: [0, ∞] (higher = more trustworthy)
- **Best for**: Encouraging model exploration

```bash
python train.py trust=entropy
```

### 3. Reputation-Based Trust
- **Principle**: Historical performance tracking and scoring
- **Range**: [0, 1] (higher = more trustworthy)
- **Best for**: Long-term client reliability assessment

```bash
python train.py trust=reputation
```

### 4. Hybrid Trust (Recommended)
- **Principle**: Weighted combination of all trust mechanisms
- **Weights**: Cosine (0.4) + Entropy (0.3) + Reputation (0.3)
- **Best for**: Comprehensive trust evaluation

```bash
python train.py trust=hybrid
```

## Advanced Usage

### 1. Large-Scale Simulations
  
#### High Client Count (50+ clients)
```bash
# Configure for high client count
python train.py dataset.num_clients=100 \
                env.ray.num_cpus=8 \
                env.ray.object_store_memory=2000000000 \
                federated.fraction_fit=0.1
```

#### Extended Training
```bash
# Long training simulation
python train.py federated.num_rounds=20 \
                training=multi_epoch \
                training.local_epochs=5 \
                training.learning_rate=0.0001
```

### 2. Non-IID Data Distribution

#### Dirichlet Partitioning (Realistic)
```bash
python train.py dataset.partitioning=dirichlet \
                dataset.alpha=0.1  # Lower alpha = more non-IID
```

#### Pathological Partitioning (Extreme)
```bash
python train.py dataset.partitioning=pathological \
                dataset.classes_per_client=2
```

### 3. Attack Simulation

#### Label Flipping Attack
```bash
python train.py simulation.enable_attacks=true \
                simulation.attack_type=label_flip \
                simulation.attack_fraction=0.2
```

#### Gaussian Noise Attack
```bash
python train.py simulation.enable_attacks=true \
                simulation.attack_type=gaussian_noise \
                simulation.noise_variance=0.1
```

### 4. Experiment Tracking

#### Enable TensorBoard
```bash
python train.py metrics.enable_tensorboard=true
# View results: tensorboard --logdir=outputs
```

#### Enable MLflow
```bash
python train.py metrics.enable_mlflow=true \
                metrics.experiment_name="my_experiment"
```

#### Save Detailed Metrics
```bash
python train.py metrics.save_csv=true \
                metrics.output_dir="./results" \
                metrics.save_plots=true
```

### 5. Custom Datasets

#### Prepare CSV Dataset
```bash
# 1. Place your CSV file in data/ directory
cp your_dataset.csv TRUST_MCNet_Redesigned/data/

# 2. Ensure target column is named 'label'
# 3. Run simulation
python train.py dataset=custom_csv \
                dataset.path=./data/your_dataset.csv
```

#### IoT Sensor Data Example
```bash
# For time-series IoT data
python train.py dataset=custom_csv \
                model=lstm \
                dataset.sequence_length=50 \
                training.local_epochs=3
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Ray Initialization Errors
```bash
# Error: Ray cluster startup failed
# Solution: Clear Ray cache and restart
ray stop
rm -rf /tmp/ray*
python train.py  # Restart simulation
```

#### 2. Memory Issues
```bash
# Error: Out of memory
# Solution: Reduce batch size or client count
python train.py dataset.batch_size=16 \
                dataset.num_clients=5 \
                env.ray.object_store_memory=500000000
```

#### 3. CUDA/GPU Issues
```bash
# Error: CUDA out of memory
# Solution: Use CPU or reduce model size
python train.py env=local \
                model.mlp.hidden_dims=[64,32]
```

#### 4. Dataset Loading Errors
```bash
# Error: Dataset not found
# Solution: Check file path and format
ls -la data/  # Verify files exist
python train.py dataset.path=./data/correct_filename.csv
```

#### 5. Flower Connection Issues
```bash
# Error: Flower simulation failed
# Solution: Reduce client count or increase timeouts
python train.py federated.min_available_clients=2 \
                simulation.client_timeout=120
```

### Debug Mode

#### Enable Verbose Logging
```bash
python train.py logging.level=DEBUG
```

#### Run Tests
```bash
cd TRUST_MCNet_Redesigned

# Run all tests
python -m tests

# Run specific tests
python -m tests.test_smoke
python -m tests.test_partitioning
```

#### Validate Architecture
```bash
python validate_architecture.py
```

## Performance Optimization

### 1. Hardware Optimization

#### CPU Optimization
```bash
# Maximize CPU usage
python train.py env.ray.num_cpus=8 \
                env.dataloader.num_workers=4
```

#### GPU Acceleration
```bash
# Enable GPU with optimization
python train.py env=gpu \
                training.enable_gpu_optimization=true \
                env.dataloader.pin_memory=true
```

#### Memory Management
```bash
# Optimize memory usage
python train.py env.ray.object_store_memory=1000000000 \
                training.profile_memory=true \
                dataset.batch_size=32
```

### 2. Training Optimization

#### Adaptive Learning
```bash
# Use adaptive learning rate
python train.py training.optimizer=adam \
                training.learning_rate=0.001 \
                training.weight_decay=1e-4
```

#### Efficient Communication
```bash
# Reduce communication overhead
python train.py federated.fraction_fit=0.5 \
                training.local_epochs=3 \
                strategy=fedadam
```

### 3. Scaling Guidelines

#### Small Scale (2-10 clients)
```yaml
dataset:
  num_clients: 5
  batch_size: 32
federated:
  num_rounds: 5
  fraction_fit: 1.0
env:
  ray:
    num_cpus: 2
    object_store_memory: 500MB
```

#### Medium Scale (10-50 clients)
```yaml
dataset:
  num_clients: 20
  batch_size: 64
federated:
  num_rounds: 10
  fraction_fit: 0.3
env:
  ray:
    num_cpus: 4
    object_store_memory: 1GB
```

#### Large Scale (50+ clients)
```yaml
dataset:
  num_clients: 100
  batch_size: 128
federated:
  num_rounds: 20
  fraction_fit: 0.1
env:
  ray:
    num_cpus: 8
    object_store_memory: 2GB
```

## Example Workflows

### 1. Quick Anomaly Detection Test
```bash
cd TRUST_MCNet_Redesigned

# Fast MNIST test
python train.py dataset=mnist \
                federated.num_rounds=3 \
                dataset.num_clients=5
```

### 2. IoT Device Simulation
```bash
# Simulate resource-constrained IoT devices
python train.py env=iot \
                dataset.num_clients=20 \
                training.local_epochs=2 \
                dataset.batch_size=16 \
                trust=hybrid
```

### 3. Production-Style Training
```bash
# Full-scale simulation with all features
python train.py dataset=custom_csv \
                model=lstm \
                strategy=fedadam \
                trust=hybrid \
                env=gpu \
                training=multi_epoch \
                federated.num_rounds=15 \
                metrics.enable_tensorboard=true \
                metrics.save_csv=true
```

### 4. Research Experiment
```bash
# Research with attack simulation
python train.py dataset.partitioning=dirichlet \
                dataset.alpha=0.1 \
                simulation.enable_attacks=true \
                simulation.attack_type=label_flip \
                simulation.attack_fraction=0.3 \
                trust=hybrid \
                metrics.enable_mlflow=true \
                metrics.experiment_name="attack_robustness"
```

## Results and Analysis

### Output Locations
- **Logs**: `training.log`, `simulation.log`
- **Hydra Outputs**: `outputs/YYYY-MM-DD/HH-MM-SS/`
- **Results**: `results/` directory
- **TensorBoard**: `runs/` directory
- **Plots**: Generated in output directories

### Key Metrics
- **Accuracy**: Model performance on test data
- **Loss**: Training and validation loss curves
- **Trust Scores**: Client trustworthiness over rounds
- **Communication Cost**: Data transfer and round time
- **Resource Usage**: CPU, memory, and training time

### Interpreting Results
1. **High Trust Scores** (>0.7): Reliable clients
2. **Stable Accuracy**: Good convergence
3. **Low Communication Cost**: Efficient federation
4. **Balanced Resource Usage**: Optimal configuration

---

## Quick Reference Commands

```bash
# Quick start (redesigned)
python enhanced_simulation.py

# Basic Hydra training
python train.py

# Multi-dataset simulation
python run_multi_dataset_simulation.py --clients 5 --rounds 3

# Custom configuration
python train.py dataset=custom_csv model=lstm strategy=fedadam trust=hybrid

# GPU training
python train.py env=gpu training=multi_epoch

# IoT simulation
python train.py env=iot dataset.num_clients=20

# Debug mode
python train.py logging.level=DEBUG

# Run tests
python -m tests
```

For more detailed information, refer to the individual README files in each subdirectory and the comprehensive documentation in the codebase.
