"""
Refactored experiment management module implementing core interfaces.

This module provides production-grade experiment orchestration following
SOLID principles and implementing the ExperimentInterface.
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import sys

# Ensure we can import from the parent directory
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from core.abstractions import BaseExperiment
from core.interfaces import (
    ExperimentInterface, 
    DataLoaderInterface, 
    ModelInterface, 
    StrategyInterface,
    TrustEvaluatorInterface,
    MetricsInterface
)
from core.types import ExperimentConfig, ExperimentPhase, Metrics
from core.exceptions import ExperimentError, ConfigurationError

# Import registries for component creation
import data
import models_new
import partitioning
import strategies
import trust_new
import metrics
import clients_new

# Access registries through modules
DataLoaderRegistry = data.DataLoaderRegistry
ModelRegistry = models_new.ModelRegistry
PartitionerRegistry = partitioning.PartitioningRegistry
StrategyRegistry = strategies.StrategyRegistry
TrustEvaluatorRegistry = trust_new.TrustEvaluatorRegistry
MetricsRegistry = metrics.MetricsCollectorRegistry
ClientRegistry = clients_new.ClientRegistry

logger = logging.getLogger(__name__)


class FederatedExperiment(BaseExperiment):
    """
    Production-grade federated learning experiment manager.
    
    Features:
    - Complete experiment lifecycle management
    - Proper resource cleanup and error handling
    - Configurable component selection via registries
    - Comprehensive logging and metrics collection
    - Scalable architecture supporting 1-1000+ clients
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize federated experiment.
        
        Args:
            config: Experiment configuration containing all component settings
        """
        super().__init__(config)
        
        # Validate experiment-specific config
        self._validate_experiment_config()
        
        # Initialize component configurations
        self.dataset_config = self.config.get('dataset', {})
        self.model_config = self.config.get('model', {})
        self.strategy_config = self.config.get('strategy', {})
        self.trust_config = self.config.get('trust', {})
        self.federated_config = self.config.get('federated', {})
        self.env_config = self.config.get('env', {})
        
        # Component instances (initialized during setup)
        self.data_loader: Optional[DataLoaderInterface] = None
        self.global_model: Optional[ModelInterface] = None
        self.partitioner = None
        self.trust_evaluator: Optional[TrustEvaluatorInterface] = None
        self.strategy: Optional[StrategyInterface] = None
        self.metrics_manager: Optional[MetricsInterface] = None
        
        # Experiment state
        self.client_data = []
        self.experiment_start_time = None
        self.num_clients = self.dataset_config.get('num_clients', 5)
        
        self.logger.info(f"Initialized federated experiment: {self.config.get('name')}")
    
    def _validate_experiment_config(self) -> None:
        """Validate experiment-specific configuration."""
        # Check required sections
        required_sections = ['dataset', 'model', 'federated']
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required config section: {section}")
        
        # Validate federated config
        fed_config = self.config.get('federated', {})
        if fed_config.get('num_rounds', 0) < 1:
            raise ConfigurationError("num_rounds must be >= 1")
        
        # Validate dataset config
        dataset_config = self.config.get('dataset', {})
        if dataset_config.get('num_clients', 0) < 1:
            raise ConfigurationError("num_clients must be >= 1")
        
        # Check for scalability constraints
        num_clients = dataset_config.get('num_clients', 5)
        if num_clients > 1000:
            self.logger.warning(f"Large scale experiment with {num_clients} clients - ensure adequate resources")
    
    def setup(self) -> None:
        """
        Setup the experiment components.
        
        Raises:
            ExperimentError: If setup fails
        """
        try:
            self.phase = ExperimentPhase.SETTING_UP
            self.logger.info("Setting up federated experiment...")
            
            # Step 1: Initialize data loader
            self._setup_data_loader()
            
            # Step 2: Load and partition data
            self._setup_data_partitioning()
            
            # Step 3: Initialize global model
            self._setup_global_model()
            
            # Step 4: Initialize trust evaluator
            self._setup_trust_evaluator()
            
            # Step 5: Initialize strategy
            self._setup_strategy()
            
            # Step 6: Initialize metrics manager
            self._setup_metrics_manager()
            
            # Step 7: Initialize clients
            self._setup_clients()
            
            self.phase = ExperimentPhase.READY
            self.logger.info("Experiment setup completed successfully")
            
        except Exception as e:
            self.phase = ExperimentPhase.FAILED
            self.logger.error(f"Experiment setup failed: {e}")
            raise ExperimentError(f"Experiment setup failed: {e}") from e
    
    def _setup_data_loader(self) -> None:
        """Initialize and configure data loader."""
        self.logger.info("Setting up data loader...")
        
        try:
            self.data_loader = DataLoaderRegistry.create_loader(self.dataset_config)
            self.logger.info(f"Data loader created: {self.dataset_config.get('name')}")
        except Exception as e:
            raise ExperimentError(f"Data loader setup failed: {e}") from e
    
    def _setup_data_partitioning(self) -> None:
        """Load data and partition among clients."""
        self.logger.info("Loading and partitioning data...")
        
        try:
            # Load datasets
            train_data, test_data = self.data_loader.load_data()
            data_info = self.data_loader.get_data_info()
            
            self.logger.info(f"Dataset loaded: {data_info}")
            
            # Validate data size vs num_clients
            dataset_size = data_info.get('size', len(train_data))
            if dataset_size < self.num_clients:
                raise ExperimentError(
                    f"Insufficient data: {dataset_size} samples for {self.num_clients} clients"
                )
            
            # Create partitioner
            partition_config = {
                'name': self.dataset_config.get('partitioning', 'iid'),
                'num_clients': self.num_clients,
                'random_seed': self.config.get('random_seed', 42)
            }
            
            # Add strategy-specific parameters
            if partition_config['name'] == 'dirichlet':
                partition_config['alpha'] = self.dataset_config.get('dirichlet_alpha', 0.5)
            elif partition_config['name'] == 'pathological':
                partition_config['classes_per_client'] = self.dataset_config.get('classes_per_client', 2)
            
            self.partitioner = PartitionerRegistry.create_partitioner(partition_config)
            
            # Partition data
            self.client_data = self.partitioner.partition(
                train_data, 
                self.num_clients,
                **{k: v for k, v in partition_config.items() if k not in ['name', 'num_clients']}
            )
            
            # Log partition statistics
            partition_sizes = [len(subset) for subset in self.client_data]
            self.logger.info(
                f"Data partitioned: {len(self.client_data)} clients, "
                f"sizes: min={min(partition_sizes)}, max={max(partition_sizes)}, "
                f"avg={sum(partition_sizes)/len(partition_sizes):.1f}"
            )
            
            # Store test data for evaluation
            self.test_data = test_data
            self.data_info = data_info
            
        except Exception as e:
            raise ExperimentError(f"Data partitioning failed: {e}") from e
    
    def _setup_global_model(self) -> None:
        """Initialize global model."""
        self.logger.info("Setting up global model...")
        
        try:
            # Enhance model config with data info
            enhanced_model_config = self.model_config.copy()
            
            # Auto-configure input/output dimensions if not specified
            if 'input_dim' not in enhanced_model_config and self.data_info:
                if 'num_features' in self.data_info:
                    enhanced_model_config['input_dim'] = self.data_info['num_features']
                elif 'data_shape' in self.data_info:
                    shape = self.data_info['data_shape']
                    if isinstance(shape, tuple) and len(shape) > 1:
                        # Flatten image data
                        enhanced_model_config['input_dim'] = shape[1] * shape[2] if len(shape) == 3 else shape[1]
            
            if 'output_dim' not in enhanced_model_config and self.data_info:
                enhanced_model_config['output_dim'] = self.data_info.get('num_classes', 2)
            
            self.global_model = ModelRegistry.create_model(enhanced_model_config)
            self.logger.info(f"Global model created: {enhanced_model_config.get('type')}")
            
        except Exception as e:
            raise ExperimentError(f"Global model setup failed: {e}") from e
    
    def _setup_trust_evaluator(self) -> None:
        """Initialize trust evaluator."""
        if not self.trust_config.get('enabled', True):
            self.logger.info("Trust evaluation disabled")
            self.trust_evaluator = None
            return
        
        self.logger.info("Setting up trust evaluator...")
        
        try:
            trust_type = self.trust_config.get('mode', 'hybrid')
            self.trust_evaluator = TrustEvaluatorRegistry.create(trust_type, self.trust_config)
            self.logger.info(f"Trust evaluator setup completed: {trust_type}")
            
        except Exception as e:
            raise ExperimentError(f"Trust evaluator setup failed: {e}") from e
    
    def _setup_strategy(self) -> None:
        """Initialize federated learning strategy."""
        self.logger.info("Setting up federated strategy...")
        
        try:
            strategy_name = self.strategy_config.get('name', 'fedavg')
            self.strategy = StrategyRegistry.create(
                strategy_name, 
                self.strategy_config,
                trust_evaluator=self.trust_evaluator
            )
            self.logger.info(f"Strategy setup completed: {strategy_name}")
            
        except Exception as e:
            raise ExperimentError(f"Strategy setup failed: {e}") from e
    
    def _setup_metrics_manager(self) -> None:
        """Initialize metrics manager."""
        self.logger.info("Setting up metrics manager...")
        
        try:
            metrics_config = self.config.get('metrics', {})
            metrics_config['experiment_name'] = self.experiment_name
            metrics_config['output_dir'] = self.config.get('output_dir', './outputs')
            
            self.metrics_manager = MetricsRegistry.create('federated_learning', metrics_config)
            self.logger.info("Metrics manager setup completed")
            
        except Exception as e:
            raise ExperimentError(f"Metrics manager setup failed: {e}") from e
    
    def _setup_clients(self) -> None:
        """Initialize federated learning clients."""
        self.logger.info("Setting up federated clients...")
        
        try:
            self.clients = []
            client_config = self.config.get('client', {})
            client_type = client_config.get('type', 'standard')
            
            for i, client_data in enumerate(self.client_data):
                client_id = f"client_{i}"
                
                # Create client-specific data loader
                client_data_loader = DataLoaderRegistry.create(
                    self.dataset_config['name'],
                    {**self.dataset_config, 'client_data': client_data, 'test_data': self.test_data}
                )
                
                # Create client-specific model (copy of global model)
                client_model = ModelRegistry.create(self.model_config)
                client_model.set_weights(self.global_model.get_weights())
                
                # Create client
                client = ClientRegistry.create(
                    client_type,
                    client_id,
                    client_config,
                    client_model,
                    client_data_loader,
                    self.trust_evaluator
                )
                
                self.clients.append(client)
            
            self.logger.info(f"Created {len(self.clients)} federated clients")
            
        except Exception as e:
            raise ExperimentError(f"Client setup failed: {e}") from e
    
    def run(self) -> Dict[str, Any]:
        """
        Run the federated learning experiment.
        
        Returns:
            Experiment results
        """
        try:
            self.phase = ExperimentPhase.RUNNING
            self.experiment_start_time = time.time()
            
            self.logger.info("Starting federated learning experiment...")
            self.logger.info(f"Configuration: {self.num_clients} clients, "
                           f"{self.federated_config.get('num_rounds')} rounds")
            
            # Run federated learning rounds
            results = self._run_federated_rounds()
            
            # Calculate total experiment time
            total_time = time.time() - self.experiment_start_time
            
            # Compile final results
            final_results = {
                'experiment_name': self.config.get('name'),
                'total_time': total_time,
                'num_clients': self.num_clients,
                'num_rounds': self.federated_config.get('num_rounds'),
                'dataset_info': self.data_info,
                'final_metrics': results.get('final_metrics', {}),
                'round_metrics': results.get('round_metrics', []),
                'config': self.config
            }
            
            # Log experiment results
            self.log_result('total_time', total_time)
            self.log_result('final_metrics', final_results['final_metrics'])
            
            self.phase = ExperimentPhase.COMPLETED
            self.logger.info(f"Experiment completed successfully in {total_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.phase = ExperimentPhase.FAILED
            self.logger.error(f"Experiment execution failed: {e}")
            self.logger.error(traceback.format_exc())
            raise ExperimentError(f"Experiment execution failed: {e}") from e
    
    def _run_federated_rounds(self) -> Dict[str, Any]:
        """
        Execute federated learning rounds.
        
        Returns:
            Round-by-round results
        """
        num_rounds = self.federated_config.get('num_rounds', 3)
        round_metrics = []
        global_weights = self.global_model.get_weights()
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"Starting round {round_num}/{num_rounds}")
            
            round_start_time = time.time()
            
            # Execute round
            round_result = self._execute_round(round_num, global_weights)
            global_weights = round_result['updated_weights']
            
            round_time = time.time() - round_start_time
            round_result['round_time'] = round_time
            
            round_metrics.append(round_result)
            
            # Record metrics
            if self.metrics_manager:
                self.metrics_manager.record_round_metrics(round_num, round_result['metrics'])
            
            self.logger.info(f"Round {round_num} completed in {round_time:.2f}s, "
                           f"metrics: {round_result.get('metrics', {})}")
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics(round_metrics)
        
        return {
            'round_metrics': round_metrics,
            'final_metrics': final_metrics,
            'updated_weights': global_weights
        }
    
    def _execute_round(self, round_num: int, global_weights: Any) -> Dict[str, Any]:
        """
        Execute a single federated learning round.
        
        Args:
            round_num: Current round number
            global_weights: Current global model weights
            
        Returns:
            Round results
        """
        try:
            # Configure round
            fit_config = self.strategy.configure_fit(round_num)
            eval_config = self.strategy.configure_evaluate(round_num)
            
            # Client selection and training
            selected_clients = self._select_clients(round_num)
            
            # Fit phase - train on selected clients
            fit_results = []
            for client in selected_clients:
                try:
                    result = client.fit(global_weights, fit_config)
                    fit_results.append(result)
                    
                    # Record client metrics
                    if self.metrics_manager:
                        self.metrics_manager.record_client_metrics(
                            client.client_id, 
                            result.get('metrics', {}),
                            round_num
                        )
                except Exception as e:
                    self.logger.warning(f"Client {client.client_id} fit failed: {e}")
                    continue
            
            if not fit_results:
                raise ExperimentError(f"No clients completed training in round {round_num}")
            
            # Aggregate updates
            aggregation_result = self.strategy.aggregate_fit(round_num, fit_results)
            updated_weights = aggregation_result['weights']
            
            # Update global model
            self.global_model.set_weights(updated_weights)
            
            # Update trust evaluator global state
            if self.trust_evaluator:
                self.trust_evaluator.update_global_state(updated_weights, round_num)
            
            # Evaluation phase
            eval_results = []
            for client in selected_clients:
                try:
                    result = client.evaluate(updated_weights, eval_config)
                    eval_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Client {client.client_id} evaluation failed: {e}")
                    continue
            
            # Aggregate evaluation results
            eval_metrics = self.strategy.aggregate_evaluate(round_num, eval_results) if eval_results else {}
            
            return {
                'round': round_num,
                'num_clients_fit': len(fit_results),
                'num_clients_eval': len(eval_results),
                'updated_weights': updated_weights,
                'metrics': {
                    'loss': eval_metrics.get('loss', float('inf')),
                    'accuracy': eval_metrics.get('accuracy', 0.0),
                    'num_clients_fit': len(fit_results),
                    'num_clients_eval': len(eval_results),
                    **aggregation_result.get('metrics', {})
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in round {round_num}: {e}")
            raise ExperimentError(f"Round {round_num} failed: {e}") from e
    
    def _select_clients(self, round_num: int) -> List[Any]:
        """
        Select clients for the current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected clients
        """
        # For now, select all clients (can be enhanced with sampling strategies)
        fraction_fit = self.strategy_config.get('fraction_fit', 1.0)
        num_selected = max(1, int(len(self.clients) * fraction_fit))
        
        # Simple random selection (can be enhanced with trust-based selection)
        import random
        selected = random.sample(self.clients, min(num_selected, len(self.clients)))
        
        self.logger.debug(f"Selected {len(selected)}/{len(self.clients)} clients for round {round_num}")
        return selected
    
    def _compute_final_metrics(self, round_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute final experiment metrics.
        
        Args:
            round_metrics: List of round metrics
            
        Returns:
            Final metrics summary
        """
        if not round_metrics:
            return {}
        
        # Extract metrics over rounds
        losses = [r['metrics'].get('loss', float('inf')) for r in round_metrics]
        accuracies = [r['metrics'].get('accuracy', 0.0) for r in round_metrics]
        
        final_metrics = {
            'final_loss': losses[-1] if losses else float('inf'),
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'best_loss': min(losses) if losses else float('inf'),
            'best_accuracy': max(accuracies) if accuracies else 0.0,
            'total_rounds': len(round_metrics),
            'convergence_rounds': self._detect_convergence(losses),
            'improvement': {
                'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else 0.0,
                'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
            }
        }
        
        return final_metrics
    
    def _detect_convergence(self, losses: List[float]) -> Optional[int]:
        """
        Detect at which round convergence occurred.
        
        Args:
            losses: List of loss values
            
        Returns:
            Round number where convergence detected, or None
        """
        if len(losses) < 3:
            return None
        
        # Simple convergence detection: loss doesn't improve for 3 consecutive rounds
        threshold = 0.001
        for i in range(2, len(losses)):
            if all(abs(losses[j] - losses[j-1]) < threshold for j in range(i-2, i+1)):
                return i + 1
        
        return None
        
        # Extract final metrics
        final_metrics = round_metrics[-1].get('metrics', {}) if round_metrics else {}
        
        return {
            'round_metrics': round_metrics,
            'final_metrics': final_metrics
        }
    
    def _execute_round(self, round_num: int) -> Dict[str, Any]:
        """
        Execute a single federated learning round.
        
        Args:
            round_num: Current round number
            
        Returns:
            Round results
        """
        # For now, return mock results
        # TODO: Implement actual federated round execution
        
        # Simulate round execution
        time.sleep(0.1)  # Simulate computation time
        
        # Mock metrics that improve over rounds
        base_loss = 1.0
        base_accuracy = 0.1
        
        loss = base_loss * (0.9 ** round_num) + 0.1  # Decreasing loss
        accuracy = min(0.95, base_accuracy + (round_num * 0.1))  # Increasing accuracy
        
        return {
            'round': round_num,
            'metrics': {
                'loss': loss,
                'accuracy': accuracy,
                'participants': min(self.num_clients, self.federated_config.get('max_clients_per_round', self.num_clients))
            },
            'trust_scores': {f'client_{i}': 0.8 + (i * 0.02) % 0.4 for i in range(min(10, self.num_clients))}
        }
    
    def cleanup(self) -> None:
        """
        Cleanup experiment resources.
        """
        try:
            self.logger.info("Cleaning up experiment resources...")
            
            # Cleanup components
            if self.metrics_manager:
                # TODO: Close metrics manager
                pass
            
            if self.global_model:
                # TODO: Cleanup model resources
                pass
            
            # Clear data references
            self.client_data = []
            self.test_data = None
            
            self.phase = ExperimentPhase.CLEANED_UP
            self.logger.info("Experiment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def save_results(self, output_path: Path) -> None:
        """
        Save experiment results to file.
        
        Args:
            output_path: Path to save results
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare results for serialization
            serializable_results = {
                'experiment_name': self.config.get('name'),
                'phase': self.phase.value if hasattr(self.phase, 'value') else str(self.phase),
                'results': self.results,
                'config': self.config
            }
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment.
        
        Returns:
            Experiment summary
        """
        return {
            'name': self.config.get('name'),
            'phase': self.phase,
            'num_clients': self.num_clients,
            'num_rounds': self.federated_config.get('num_rounds'),
            'dataset': self.dataset_config.get('name'),
            'model': self.model_config.get('type'),
            'strategy': self.strategy_config.get('name'),
            'partitioning': self.dataset_config.get('partitioning'),
            'results_available': bool(self.results)
        }


class ExperimentManager:
    """
    High-level experiment management and orchestration.
    
    Provides utilities for running multiple experiments, parameter sweeps,
    and experiment comparison.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize experiment manager.
        
        Args:
            base_config: Base configuration for experiments
        """
        self.base_config = base_config
        self.experiments: List[FederatedExperiment] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_experiment(self, config_overrides: Optional[Dict[str, Any]] = None) -> FederatedExperiment:
        """
        Create a new experiment with optional configuration overrides.
        
        Args:
            config_overrides: Configuration overrides
            
        Returns:
            New experiment instance
        """
        # Merge base config with overrides
        experiment_config = self.base_config.copy()
        if config_overrides:
            experiment_config.update(config_overrides)
        
        experiment = FederatedExperiment(experiment_config)
        self.experiments.append(experiment)
        
        return experiment
    
    def run_experiment(self, experiment: FederatedExperiment) -> Dict[str, Any]:
        """
        Run a single experiment with full lifecycle management.
        
        Args:
            experiment: Experiment to run
            
        Returns:
            Experiment results
        """
        try:
            experiment.setup()
            results = experiment.run()
            return results
        finally:
            experiment.cleanup()
    
    def run_parameter_sweep(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Run a parameter sweep across multiple configurations.
        
        Args:
            parameter_grid: Grid of parameters to sweep over
            
        Returns:
            List of results from all experiments
        """
        from itertools import product
        
        # Generate all parameter combinations
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        all_results = []
        
        for combination in product(*values):
            # Create config overrides for this combination
            config_overrides = dict(zip(keys, combination))
            
            self.logger.info(f"Running experiment with config: {config_overrides}")
            
            # Create and run experiment
            experiment = self.create_experiment(config_overrides)
            
            try:
                results = self.run_experiment(experiment)
                results['parameter_combination'] = config_overrides
                all_results.append(results)
                
            except Exception as e:
                self.logger.error(f"Experiment failed with config {config_overrides}: {e}")
                all_results.append({
                    'parameter_combination': config_overrides,
                    'error': str(e),
                    'failed': True
                })
        
        return all_results
    
    def compare_experiments(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results from multiple experiments.
        
        Args:
            results: List of experiment results
            
        Returns:
            Comparison summary
        """
        if not results:
            return {'error': 'No results to compare'}
        
        # Extract metrics for comparison
        comparison = {
            'num_experiments': len(results),
            'successful_experiments': len([r for r in results if not r.get('failed', False)]),
            'metrics_comparison': {}
        }
        
        # Compare final metrics
        final_metrics = []
        for result in results:
            if not result.get('failed', False) and 'final_metrics' in result:
                final_metrics.append(result['final_metrics'])
        
        if final_metrics:
            # Calculate statistics for each metric
            metric_names = set()
            for metrics in final_metrics:
                metric_names.update(metrics.keys())
            
            for metric_name in metric_names:
                values = [m.get(metric_name) for m in final_metrics if metric_name in m]
                if values and all(isinstance(v, (int, float)) for v in values):
                    comparison['metrics_comparison'][metric_name] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'count': len(values)
                    }
        
        return comparison

# Experiment Registry
class ExperimentRegistry:
    """Registry for experiment types."""
    
    _experiments = {
        'federated_learning': FederatedExperiment,
    }
    
    @classmethod
    def register(cls, name: str, experiment_class: type) -> None:
        """Register a new experiment type."""
        if not issubclass(experiment_class, ExperimentInterface):
            raise ValueError(f"Experiment {experiment_class} must implement ExperimentInterface")
        cls._experiments[name] = experiment_class
        logger.info(f"Registered experiment type: {name}")
    
    @classmethod
    def create(cls, name: str, config: ExperimentConfig) -> ExperimentInterface:
        """Create an experiment instance."""
        if name not in cls._experiments:
            available = ', '.join(cls._experiments.keys())
            raise ExperimentError(f"Experiment type '{name}' not found. Available: {available}")
        
        experiment_class = cls._experiments[name]
        return experiment_class(config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available experiment types."""
        return list(cls._experiments.keys())


def create_experiment_manager(
    config: ExperimentConfig,
    experiment_type: str = 'federated_learning'
) -> ExperimentInterface:
    """
    Factory function to create experiment manager instances.
    
    Args:
        config: Experiment configuration
        experiment_type: Type of experiment to create
        
    Returns:
        Experiment manager instance
    """
    return ExperimentRegistry.create(experiment_type, config)


# Export public interface
__all__ = [
    'FederatedExperiment',
    'MultiExperimentRunner',
    'ExperimentRegistry',
    'create_experiment_manager'
]
