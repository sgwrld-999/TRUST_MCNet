# TRUST-MCNet: Federated Learning Framework with Trust Mechanisms

A modern, production-ready federated learning framework for **IoT anomaly detection** with **advanced trust evaluation mechanisms**, built using enterprise-grade architecture patterns and best practices.

## 🚀 Project Overview

This completely redesigned TRUST-MCNet framework introduces:

- **🏗️ Modern Architecture**: Strategy and Registry patterns with dependency injection
- **⚙️ Advanced Configuration**: OmegaConf schemas with hierarchical config groups
- **🔧 Enhanced Resource Management**: Ray context managers with guaranteed cleanup
- **📊 Comprehensive Metrics**: TensorBoard/MLflow integration with federated logging
- **🧪 Robust Testing**: Full test suite with unit, integration, and smoke tests
- **💾 Memory Optimization**: Automatic GPU cache clearing and garbage collection
- **🔄 Fault Tolerance**: Retry logic, error handling, and graceful degradation
- **📈 Multi-Epoch Training**: Configurable local training with advanced client logic

## 🏛️ Architecture Overview

The framework follows **SOLID principles** and uses modern patterns:

- **Strategy Pattern**: Pluggable dataset partitioning (IID, Dirichlet, Pathological)
- **Registry Pattern**: Extensible dataset and partitioner management
- **Context Managers**: Guaranteed Ray resource cleanup and memory management
- **Configuration as Code**: OmegaConf dataclass schemas with validation
- **Dependency Injection**: Clean separation of concerns and testability

### Core Technologies

- **Hydra + OmegaConf**: Type-safe hierarchical configuration
- **Ray**: Distributed actor-based client execution
- **Flower**: Modern federated learning simulation
- **PyTorch**: Deep learning with automatic GPU management
- **TensorBoard/MLflow**: Experiment tracking and visualization

## 📁 Project Structure

```
TRUST_MCNet/
├── src/trust_mcnet/                  # Main package source code
│   ├── clients/                     # Federated learning clients
│   ├── core/                        # Core framework components
│   ├── models/                      # Neural network models
│   ├── trust_module/                # Trust evaluation mechanisms
│   ├── utils/                       # Utility functions
│   ├── strategies/                  # FL strategies and algorithms
│   ├── partitioning/               # Data partitioning methods
│   ├── metrics/                    # Evaluation metrics
│   └── explainability/            # Model explanation tools
├── config/                          # Configuration management
│   ├── schemas.py                   # OmegaConf dataclass schemas
│   ├── training/                    # Training configurations
│   ├── dataset/                     # Dataset configurations
│   └── ray/                         # Ray cluster configurations
├── data/                            # Datasets and data processing
├── examples/                        # Usage examples and demos
│   └── start_simulation.py          # Main simulation script
├── scripts/                         # Experiment scripts
├── tests/                           # Test suite
├── docs/                            # Documentation
│   ├── api/                         # API documentation
│   ├── reports/                     # Analysis reports
│   └── diagrams/                    # Architecture diagrams
├── logs/                            # Runtime logs
├── outputs/                         # Experiment outputs
├── results/                         # Simulation results
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```
│   │   └── custom_csv.yaml
│   ├── env/                         # Environment configurations
│   │   ├── local.yaml
│   │   ├── iot.yaml
│   │   └── gpu.yaml
│   ├── strategy/                    # FL strategy configurations
│   │   ├── fedavg.yaml
│   │   ├── fedadam.yaml
│   │   └── fedprox.yaml
│   ├── trust/                       # Trust mechanism configurations
│   │   ├── hybrid.yaml
│   │   ├── cosine.yaml
│   │   ├── entropy.yaml
│   │   └── reputation.yaml
│   └── model/                       # Model architecture configurations
│       ├── mlp.yaml
│       └── lstm.yaml
├── clients/                          # Client implementations
│   ├── enhanced_ray_client.py       # Enhanced client with error handling
│   └── ray_flwr_client.py          # Original Ray Flower client
├── utils/                           # Utility modules
│   ├── partitioning.py             # Strategy pattern for data partitioning
│   ├── dataset_registry.py         # Registry pattern for dataset management
│   ├── ray_utils.py                # Ray context managers and utilities
│   ├── metrics_logger.py           # Federated metrics logging system
│   └── data_utils.py               # Core data utilities
├── models/                          # Model definitions
│   └── model.py                    # MLP and LSTM implementations
├── trust_module/                    # Trust evaluation system
│   └── trust_evaluator.py         # Multi-modal trust mechanisms
├── tests/                           # Comprehensive test suite
│   ├── __init__.py                 # Test runner and utilities
│   ├── test_partitioning.py        # Partitioner strategy tests
│   ├── test_dataset_registry.py    # Dataset registry tests
│   ├── test_models.py              # Model architecture tests
│   └── test_smoke.py               # End-to-end smoke tests
├── enhanced_simulation.py          # New orchestrator with all patterns
├── simulation.py                    # Original simulation logic
├── train.py                         # Hydra entry point
├── examples.py                      # Usage examples and demos
├── requirements.txt                 # Production dependencies
├── pyproject.toml                   # Modern packaging and dev tools
└── README.md                        # This documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+ (recommended: 3.9+)
- PyTorch 1.12+ with CUDA support (optional)
- 8GB+ RAM for medium datasets
- CUDA-compatible GPU (optional, for acceleration)

### Quick Installation

```bash
# Clone and navigate to the redesigned directory
cd TRUST_MCNet_Redesigned

# Install all dependencies (including dev tools)
pip install -r requirements.txt

# Alternative: Install in development mode with all extras
pip install -e .[dev,docs,experiment]

# Verify installation by running tests
python -m tests

# Quick smoke test
python enhanced_simulation.py --config-name=config dataset=mnist
```

### Data Preparation
- **MNIST**: Downloads automatically on first run
- **Custom CSV**: Place files in `data/` directory with target column named 'label'
- **Validation**: All datasets undergo automatic preprocessing and validation

## 🚀 Usage Examples

### Basic Training

```bash
# Enhanced simulation with all new features (RECOMMENDED)
python enhanced_simulation.py

# Original simulation (legacy compatibility)
python train.py

# Override any configuration group
python enhanced_simulation.py dataset=custom_csv model=lstm training=multi_epoch
```

### Advanced Configuration

```bash
# Multi-epoch training with enhanced logging
python enhanced_simulation.py training=multi_epoch ray=distributed

# GPU training with memory optimization
python enhanced_simulation.py env=gpu training.enable_gpu_optimization=true

# IoT edge deployment simulation
python enhanced_simulation.py env=iot dataset.num_clients=10 training.local_epochs=3

# Custom partitioning strategies
python enhanced_simulation.py dataset.partitioning=dirichlet dataset.alpha=0.5
```

### Experiment Tracking & Monitoring

```bash
# Enable TensorBoard logging
python enhanced_simulation.py metrics.enable_tensorboard=true

# Full experiment tracking with MLflow
python enhanced_simulation.py metrics.enable_mlflow=true metrics.experiment_name="my_experiment"

# Export detailed metrics to CSV
python enhanced_simulation.py metrics.save_csv=true metrics.output_dir="./results"
```

### Testing & Validation

```bash
# Run full test suite
python -m tests

# Run specific test categories
python -m tests.test_partitioning  # Test partitioning strategies
python -m tests.test_models        # Test model architectures
python -m tests.test_smoke         # End-to-end smoke tests

# Performance profiling
python enhanced_simulation.py training.profile_memory=true training.profile_compute=true
```


## ⚙️ Configuration System

The new configuration system uses **OmegaConf dataclass schemas** for type safety and validation:

### Configuration Schema (`config/schemas.py`)

```python
@dataclass
class TrainingConfig:
    local_epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    enable_gpu_optimization: bool = True
    max_retries: int = 3
    
@dataclass  
class DatasetConfig:
    name: str = "mnist"
    num_clients: int = 5
    partitioning: str = "iid"  # iid, dirichlet, pathological
    alpha: float = 0.5         # For Dirichlet partitioning
    
@dataclass
class MetricsConfig:
    enable_tensorboard: bool = False
    enable_mlflow: bool = False
    save_csv: bool = True
    experiment_name: str = "federated_experiment"
```

### Main Configuration (`config/config.yaml`)

```yaml
defaults:
  - training: default      # or multi_epoch
  - ray: local            # or distributed  
  - dataset: mnist        # or custom_csv
  - env: local           # local, gpu, iot
  - strategy: fedavg     # fedavg, fedadam, fedprox
  - trust: hybrid        # hybrid, cosine, entropy, reputation
  - model: mlp           # mlp, lstm
  - _self_              # Include this config

# Global federated learning settings
federated:
  num_rounds: 3
  fraction_fit: 0.8
  fraction_evaluate: 0.2
  min_fit_clients: 2
  min_evaluate_clients: 1
  min_available_clients: 2

# Enhanced simulation settings  
simulation:
  use_enhanced_client: true
  enable_trust_evaluation: true
  trust_threshold: 0.5
  enable_metrics_aggregation: true
```

### Training Configurations

**Single Epoch Training** (`config/training/default.yaml`)
```yaml
training:
  local_epochs: 1
  learning_rate: 0.001
  batch_size: 32
  enable_gpu_optimization: true
  enable_memory_cleanup: true
  max_retries: 3
  retry_delay: 1.0
```

**Multi-Epoch Training** (`config/training/multi_epoch.yaml`)
```yaml
training:
  local_epochs: 5
  learning_rate: 0.01
  batch_size: 64
  enable_gpu_optimization: true
  enable_memory_cleanup: true
  gradient_clipping: 1.0
  max_retries: 5
  retry_delay: 2.0
```

### Ray Configurations

**Local Development** (`config/ray/local.yaml`)
```yaml
ray:
  num_cpus: 4
  num_gpus: 0
  memory_limit: "2GB"
  object_store_memory: 1000000000
  enable_logging: true
  dashboard_port: 8265
```

**Distributed Setup** (`config/ray/distributed.yaml`)  
```yaml
ray:
  num_cpus: 16
  num_gpus: 2
  memory_limit: "8GB"
  object_store_memory: 4000000000
  enable_logging: true
  dashboard_port: 8265
  cluster_mode: true
```

## 🏗️ Architecture Deep Dive

### 1. Strategy Pattern for Dataset Partitioning

**Registry-based extensible partitioning** (`utils/partitioning.py`):

```python
@PartitionerRegistry.register("iid")
class IIDPartitioner(BasePartitioner):
    def partition(self, dataset: Dataset, num_clients: int) -> Dict[int, Indices]:
        # Uniform random distribution
        
@PartitionerRegistry.register("dirichlet")  
class DirichletPartitioner(BasePartitioner):
    def partition(self, dataset: Dataset, num_clients: int, alpha: float = 0.5) -> Dict[int, Indices]:
        # Dirichlet distribution for non-IID data
        
@PartitionerRegistry.register("pathological")
class PathologicalPartitioner(BasePartitioner):
    def partition(self, dataset: Dataset, num_clients: int, shards_per_client: int = 2) -> Dict[int, Indices]:
        # Pathological non-IID (few classes per client)
```

### 2. Registry Pattern for Dataset Management

**Decoupled data loading** (`utils/dataset_registry.py`):

```python
@DatasetRegistry.register("mnist")
class MNISTLoader(BaseDatasetLoader):
    def load_dataset(self, config: DatasetConfig) -> Tuple[Dataset, Dataset]:
        # MNIST loading with transforms
        
@DatasetRegistry.register("csv")
class CSVLoader(BaseDatasetLoader):
    def load_dataset(self, config: DatasetConfig) -> Tuple[Dataset, Dataset]:
        # CSV loading with validation
        
class DataManager:
    def __init__(self, partitioner_registry: PartitionerRegistry, dataset_registry: DatasetRegistry):
        # Dependency injection for clean separation
```

### 3. Ray Context Management

**Guaranteed resource cleanup** (`utils/ray_utils.py`):

```python
class RayContextManager:
    def __enter__(self) -> "RayContextManager":
        ray.init(**self.config)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Guaranteed cleanup regardless of errors
        self.cleanup_memory()
        ray.shutdown()
        
    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

### 4. Enhanced Client Logic

**Robust error handling and retry logic** (`clients/enhanced_ray_client.py`):

```python
@ray.remote
class EnhancedRayFlowerClient:
    def fit_with_retry(self, parameters, config):
        for attempt in range(self.max_retries):
            try:
                return self._fit_internal(parameters, config)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                self.logger.warning(f"Fit attempt {attempt + 1} failed: {e}")
                time.sleep(self.retry_delay)
                
    def _cleanup_memory(self):
        """Memory and resource cleanup after training/evaluation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

### 5. Federated Metrics Logging

**Comprehensive experiment tracking** (`utils/metrics_logger.py`):

```python
class FederatedMetricsLogger:
    def __init__(self, config: MetricsConfig):
        if config.enable_tensorboard:
            self.tb_writer = SummaryWriter()
        if config.enable_mlflow:
            mlflow.start_run(run_name=config.experiment_name)
            
    def log_federated_metrics(self, round_num: int, metrics: Dict[str, float]):
        # Log to all enabled backends (TensorBoard, MLflow, CSV, JSON)
        
    def aggregate_client_metrics(self, client_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        # Compute federated averages, std dev, min/max across clients
```

## 🧪 Testing Framework

### Comprehensive Test Suite

The framework includes **4 test categories** for robustness:

```bash
# Run all tests
python -m tests

# Individual test categories
python -c "from tests.test_partitioning import *; test_all_partitioners()"
python -c "from tests.test_dataset_registry import *; test_csv_dataset_loading()"  
python -c "from tests.test_models import *; test_mlp_forward_pass()"
python -c "from tests.test_smoke import *; test_enhanced_simulation_smoke()"
```

### Test Categories

1. **Partitioning Tests** (`tests/test_partitioning.py`)
   - All partitioner strategies (IID, Dirichlet, Pathological)
   - Registry pattern functionality
   - Edge cases and error handling

2. **Dataset Registry Tests** (`tests/test_dataset_registry.py`)
   - Dataset loading and validation
   - CSV format handling
   - Registry extensibility

3. **Model Tests** (`tests/test_models.py`)
   - Forward pass validation
   - Gradient computation
   - Robustness to different input shapes

4. **Smoke Tests** (`tests/test_smoke.py`)
   - End-to-end simulation workflows
   - Configuration loading
   - Integration testing

## 🚀 Performance & Scaling

### Memory Optimization

- **Automatic GPU Cache Clearing**: `torch.cuda.empty_cache()` after training/evaluation
- **Garbage Collection**: Explicit `gc.collect()` calls for memory hygiene
- **Configurable DataLoader**: Tunable `num_workers`, `pin_memory`, `prefetch_factor`
- **Resource Monitoring**: Built-in memory and compute profiling

### Ray Parallelism

- **Actor-Based Clients**: True parallelism for independent training tasks
- **Resource Isolation**: CPU/GPU/memory limits per client
- **Fault Tolerance**: Automatic retry logic with exponential backoff
- **Dashboard Monitoring**: Real-time cluster resource visualization

### Scalability Features

```python
# Scale to 100+ clients with distributed Ray
python enhanced_simulation.py ray=distributed dataset.num_clients=100

# Edge deployment simulation
python enhanced_simulation.py env=iot dataset.num_clients=50 training.local_epochs=10

# Memory-constrained environments
python enhanced_simulation.py training.batch_size=16 ray.memory_limit="1GB"
```

## 🔧 Extending the Framework

### Adding New Dataset Loaders

1. **Create dataset config** (`config/dataset/my_dataset.yaml`):
```yaml
dataset:
  name: my_dataset
  path: "./data/my_data"
  num_clients: 10
  partitioning: dirichlet
  alpha: 0.3
```

2. **Register dataset loader** (`utils/dataset_registry.py`):
```python
@DatasetRegistry.register("my_dataset")
class MyDatasetLoader(BaseDatasetLoader):
    def load_dataset(self, config: DatasetConfig) -> Tuple[Dataset, Dataset]:
        # Implement custom loading logic
        return train_dataset, test_dataset
```

3. **Use new dataset**:
```bash
python enhanced_simulation.py dataset=my_dataset
```

### Adding New Partitioning Strategies

1. **Register partitioner** (`utils/partitioning.py`):
```python
@PartitionerRegistry.register("my_partitioner")
class MyPartitioner(BasePartitioner):
    def partition(self, dataset: Dataset, num_clients: int, **kwargs) -> Dict[int, Indices]:
        # Implement custom partitioning logic
        return client_indices
```

2. **Use new partitioner**:
```bash
python enhanced_simulation.py dataset.partitioning=my_partitioner
```

### Adding New Trust Mechanisms

1. **Create trust config** (`config/trust/my_trust.yaml`):
```yaml
trust:
  mechanism: my_trust
  threshold: 0.6
  custom_param: 0.8
```

2. **Extend trust evaluator** (`trust_module/trust_evaluator.py`):
```python
def evaluate_trust_my_trust(self, local_weights, global_weights, **kwargs):
    # Implement custom trust evaluation
    return trust_score
```

## 🔍 Monitoring & Debugging

### Experiment Tracking

**TensorBoard Integration**:
```bash
# Enable TensorBoard logging
python enhanced_simulation.py metrics.enable_tensorboard=true

# View logs (in separate terminal)
tensorboard --logdir=./outputs/tensorboard
```

**MLflow Integration**:
```bash
# Enable MLflow tracking
python enhanced_simulation.py metrics.enable_mlflow=true metrics.experiment_name="my_experiment"

# View MLflow UI
mlflow ui
```

### Log Analysis

**Structured Logging Output**:
```
2024-01-15 10:30:15 | INFO | Round 1/3 | Fit clients: 4/5 selected
2024-01-15 10:30:16 | INFO | Client 0 | Local epochs: 3 | Loss: 0.045 | Accuracy: 0.892
2024-01-15 10:30:17 | WARNING | Client 2 | Retry attempt 2/3 | Error: CUDA out of memory
2024-01-15 10:30:18 | INFO | Trust Evaluation | Client 0: 0.85 | Client 1: 0.92 | Client 3: 0.78
2024-01-15 10:30:19 | INFO | Federated Metrics | Global Loss: 0.052 | Global Accuracy: 0.887
```

**Performance Profiling**:
```bash
# Enable detailed profiling
python enhanced_simulation.py training.profile_memory=true training.profile_compute=true

# Monitor Ray dashboard
open http://localhost:8265
```

## 📊 Results & Metrics

### Federated Learning Metrics

- **Client-Level**: Loss, accuracy, training time, memory usage per client
- **Global Level**: Aggregated metrics across all clients per round
- **Trust Metrics**: Trust scores, client selection ratios, anomaly detection rates
- **System Metrics**: Ray resource utilization, memory consumption, fault rates

### Export Formats

- **CSV**: Detailed metrics for analysis (`metrics.save_csv=true`)
- **JSON**: Structured data for programmatic access
- **TensorBoard**: Interactive visualization and comparison
- **MLflow**: Experiment management and model versioning

## 🛠️ Development & Contributing

### Development Setup

```bash
# Install with development dependencies
pip install -e .[dev,docs,experiment]

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black . --check
flake8 .
mypy .

# Generate documentation
sphinx-build -b html docs/ docs/_build/
```

### Code Quality Standards

- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for all public methods
- **Error Handling**: Comprehensive exception handling with logging
- **Testing**: Minimum 80% code coverage with meaningful tests
- **Configuration**: All behavior must be configurable via OmegaConf schemas

### Architecture Principles

1. **SOLID Principles**: Single responsibility, open/closed, dependency inversion
2. **Design Patterns**: Strategy, Registry, Context Manager, Dependency Injection
3. **Clean Code**: Readable, maintainable, well-documented code
4. **Error Resilience**: Graceful degradation and comprehensive error handling
5. **Performance**: Memory optimization, resource cleanup, and scalability

## 📄 License

This project maintains the same license as the original TRUST-MCNet framework.

## 🙏 Acknowledgments

- **Original TRUST-MCNet**: Foundation trust mechanisms and anomaly detection logic
- **Flower Framework**: Modern federated learning simulation capabilities  
- **Ray Framework**: Distributed computing and actor-based parallelism
- **Hydra/OmegaConf**: Advanced configuration management and validation

---

**Ready to get started?** Try the enhanced simulation:

```bash
cd TRUST_MCNet_Redesigned
pip install -r requirements.txt
python enhanced_simulation.py
```

For questions, issues, or contributions, please follow the development guidelines above and maintain backward compatibility with existing configurations.
