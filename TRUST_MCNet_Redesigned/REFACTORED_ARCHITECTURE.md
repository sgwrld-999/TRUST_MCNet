# TRUST-MCNet Refactored Architecture

## Overview

This document describes the refactored TRUST-MCNet codebase that follows SOLID principles and software engineering best practices. The new architecture is scalable, production-grade, reliable, robust, and flexible, supporting 1 to 1000+ clients via configuration.

## Architecture Overview

The refactored codebase follows a clean, interface-based architecture with the following key principles:

### 🏗️ Core Architecture Components

#### 1. **Core Module** (`/core/`)
- **`interfaces.py`**: Defines protocols for all major components
- **`abstractions.py`**: Provides base classes implementing common functionality
- **`exceptions.py`**: Custom exception hierarchy for robust error handling
- **`types.py`**: Type aliases and enums for type safety

#### 2. **Component Modules**
- **`/data/`**: Data loading with registry pattern
- **`/models_new/`**: Model implementations with interface compliance
- **`/partitioning/`**: Data partitioning strategies
- **`/strategies/`**: Federated learning strategies (FedAvg, FedAdam, etc.)
- **`/trust_new/`**: Trust evaluation mechanisms
- **`/metrics/`**: Comprehensive metrics collection
- **`/clients_new/`**: Federated learning clients
- **`/experiments/`**: Experiment orchestration and management

### 🔧 Key Design Patterns

1. **Interface Segregation**: Small, focused interfaces for each component
2. **Dependency Injection**: Components receive dependencies via constructor
3. **Registry Pattern**: Dynamic component selection and registration
4. **Factory Pattern**: Centralized object creation
5. **Strategy Pattern**: Pluggable algorithms (partitioning, trust, etc.)
6. **Template Method**: Common workflows in base classes

## Interface Definitions

### Core Interfaces

```python
# Data Loading
class DataLoaderInterface(Protocol):
    def load_data(self) -> Tuple[Any, Any]
    def get_train_loader(self, batch_size: int) -> Any
    def get_test_loader(self, batch_size: int) -> Any
    def get_data_info(self) -> Dict[str, Any]

# Model Interface
class ModelInterface(Protocol):
    def get_weights(self) -> ModelWeights
    def set_weights(self, weights: ModelWeights) -> None
    def train(self) -> None
    def eval(self) -> None
    def compute_loss(self, batch: Any) -> Any
    def predict(self, batch: Any) -> Any

# Strategy Interface
class StrategyInterface(Protocol):
    def configure_fit(self, round_num: int, **kwargs) -> Dict[str, Any]
    def configure_evaluate(self, round_num: int, **kwargs) -> Dict[str, Any]
    def aggregate_fit(self, round_num: int, results: List[ClientResults], **kwargs) -> AggregationResult
    def aggregate_evaluate(self, round_num: int, results: List[ClientResults], **kwargs) -> Dict[str, Any]

# Trust Evaluator Interface
class TrustEvaluatorInterface(Protocol):
    def evaluate_trust(self, client_id: Optional[str] = None, **kwargs) -> TrustScore
    def update_global_state(self, global_update: Any, round_num: int, **kwargs) -> None
    def get_trust_metrics(self, client_id: Optional[str] = None) -> TrustMetrics

# Client Interface
class ClientInterface(Protocol):
    def fit(self, parameters: ModelWeights, config: Dict[str, Any], **kwargs) -> TrainingResults
    def evaluate(self, parameters: ModelWeights, config: Dict[str, Any], **kwargs) -> EvaluationResults
    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]

# Experiment Interface
class ExperimentInterface(Protocol):
    def setup(self) -> None
    def run(self) -> Dict[str, Any]
    def cleanup(self) -> None
    def get_results(self) -> Dict[str, Any]
```

## Registry Pattern Implementation

Each component type has a registry for dynamic selection:

```python
# Example: Data Loader Registry
class DataLoaderRegistry:
    _loaders = {
        'mnist': MNISTDataLoader,
        'csv': CSVDataLoader,
    }
    
    @classmethod
    def register(cls, name: str, loader_class: type) -> None:
        cls._loaders[name] = loader_class
    
    @classmethod
    def create(cls, name: str, config: ConfigType) -> DataLoaderInterface:
        if name not in cls._loaders:
            raise DataLoadingError(f"Data loader '{name}' not found")
        return cls._loaders[name](config)
```

## Configuration-Driven Scaling

### Flexible Client Scaling (1-1000+ clients)

```yaml
# config/dataset/scalable_mnist.yaml
name: mnist
num_clients: 100  # Easily scale from 1 to 1000+
partitioning_strategy: dirichlet
partitioning_params:
  alpha: 0.5
batch_size: 32
```

### Flexible Label Mapping

```yaml
# config/dataset/iot_general.yaml
name: iot_general
label_mapping:
  normal: 0
  anomaly: 1
  attack: 2
  suspicious: 3
# No hardcoded binary logic - fully configurable
```

### Modular Strategy Selection

```yaml
# config/strategy/fedadam.yaml
name: fedadam
server_learning_rate: 1.0
beta_1: 0.9
beta_2: 0.999
epsilon: 1e-8
fraction_fit: 1.0
min_fit_clients: 2
```

## Production-Grade Features

### 🛡️ Robust Error Handling

```python
# Custom exception hierarchy
class TRUSTMCNetError(Exception): pass
class ConfigurationError(TRUSTMCNetError): pass
class DataLoadingError(TRUSTMCNetError): pass
class ModelError(TRUSTMCNetError): pass
class StrategyError(TRUSTMCNetError): pass
class TrustEvaluationError(TRUSTMCNetError): pass
class ExperimentError(TRUSTMCNetError): pass
class ClientError(TRUSTMCNetError): pass
class MetricsError(TRUSTMCNetError): pass
```

### 📊 Comprehensive Metrics

```python
# Production-grade metrics collection
class FederatedLearningMetrics(BaseMetrics):
    def record_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None
    def record_client_metrics(self, client_id: str, metrics: Dict[str, Any]) -> None
    def record_trust_metrics(self, trust_data: Dict[str, Any]) -> None
    def get_summary(self, include_history: bool = False) -> MetricsSnapshot
    def save(self, format: List[str] = ['json', 'csv']) -> None
```

### 🔍 Trust Evaluation

```python
# Multiple trust evaluation strategies
class TrustEvaluatorRegistry:
    _evaluators = {
        'cosine': CosineSimilarityTrustEvaluator,
        'entropy': EntropyBasedTrustEvaluator,
        'hybrid': HybridTrustEvaluator,
    }
```

## Usage Examples

### Basic Training

```python
# Using the new refactored architecture
python train_refactored.py
```

### Scaled Training (100 clients)

```python
python train_refactored.py dataset.num_clients=100
```

### Custom Configuration

```python
python train_refactored.py \
    dataset=mnist \
    model=mlp \
    strategy=fedadam \
    trust=hybrid \
    dataset.num_clients=50 \
    federated.num_rounds=20
```

### Programmatic Usage

```python
from experiments import create_experiment_manager

# Create experiment configuration
config = {
    'dataset': {'name': 'mnist', 'num_clients': 100},
    'model': {'type': 'mlp', 'hidden_dims': [128, 64]},
    'strategy': {'name': 'fedavg'},
    'trust': {'mode': 'hybrid', 'enabled': True},
    'federated': {'num_rounds': 10}
}

# Create and run experiment
experiment = create_experiment_manager(config, 'federated_learning')
experiment.setup()
results = experiment.run()
experiment.cleanup()
```

## Extensibility

### Adding New Components

#### 1. New Data Loader
```python
class CustomDataLoader(BaseDataLoader):
    def load_data(self) -> Tuple[Any, Any]:
        # Implementation
        pass

# Register
DataLoaderRegistry.register('custom', CustomDataLoader)
```

#### 2. New Model
```python
class TransformerModel(BaseModel):
    def __init__(self, config: ConfigType):
        super().__init__(config)
        # Implementation

# Register
ModelRegistry.register('transformer', TransformerModel)
```

#### 3. New Strategy
```python
class FedProxStrategy(BaseStrategy):
    def aggregate_fit(self, round_num: int, results: List[ClientResults]) -> AggregationResult:
        # FedProx implementation
        pass

# Register
StrategyRegistry.register('fedprox', FedProxStrategy)
```

## Testing

### Component Testing
```python
# Test individual components
pytest tests/test_data_loaders.py
pytest tests/test_models.py
pytest tests/test_strategies.py
pytest tests/test_trust_evaluators.py
```

### Integration Testing
```python
# Test end-to-end workflows
pytest tests/test_experiment_integration.py
```

### Scalability Testing
```python
# Test with various client counts
pytest tests/test_scalability.py -k "test_100_clients"
pytest tests/test_scalability.py -k "test_1000_clients"
```

## Performance Considerations

### Memory Management
- Automatic resource cleanup in base classes
- Configurable batch sizes and data loading
- Optional client sampling for large-scale scenarios

### Computational Efficiency
- Lazy loading of components
- Efficient data partitioning algorithms
- Optimized aggregation strategies

### Scalability
- Ray integration for distributed execution
- Configurable resource allocation
- Memory-efficient client management

## Migration Guide

### From Legacy Code

1. **Update imports**: Use new module structure
2. **Update configuration**: Use YAML-based config
3. **Update component creation**: Use registry pattern
4. **Update error handling**: Use new exception hierarchy

### Backward Compatibility

The refactored code maintains compatibility with existing configurations through adapter patterns where necessary.

## Configuration Reference

### Complete Configuration Schema

```yaml
# Main configuration file
dataset:
  name: mnist  # or csv, iot_general
  num_clients: 10
  partitioning_strategy: iid  # or dirichlet, pathological
  batch_size: 32

model:
  type: mlp  # or lstm
  input_dim: 784
  output_dim: 10
  hidden_dims: [128, 64]

strategy:
  name: fedavg  # or fedadam
  fraction_fit: 1.0
  min_fit_clients: 2

trust:
  mode: hybrid  # or cosine, entropy
  enabled: true
  threshold: 0.5

federated:
  num_rounds: 10

client:
  type: standard  # or ray
  local_epochs: 1
  learning_rate: 0.01
  optimizer: sgd

metrics:
  save_format: [json, csv]
  save_frequency: round

experiment:
  name: my_experiment
  output_dir: ./outputs
```

## Conclusion

The refactored TRUST-MCNet architecture provides:

✅ **SOLID Compliance**: Interface segregation, dependency injection, single responsibility
✅ **Scalability**: Support for 1-1000+ clients via configuration
✅ **Extensibility**: Registry patterns for easy component addition
✅ **Reliability**: Comprehensive error handling and resource management  
✅ **Flexibility**: Configuration-driven behavior without hardcoded logic
✅ **Production-Ready**: Comprehensive metrics, logging, and testing
✅ **Maintainability**: Clean interfaces and modular design

The new architecture enables easy experimentation, scaling, and extension while maintaining production-grade reliability and performance.
