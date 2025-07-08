"""
Smoke tests for federated learning simulation.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFederatedLearningSmoke(unittest.TestCase):
    """Smoke tests for federated learning components."""
    
    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            'dataset': {
                'name': 'mnist',
                'path': './test_data',
                'num_clients': 2,
                'eval_fraction': 0.2,
                'batch_size': 16,
                'partitioning': 'iid',
                'transforms': {'normalize': True}
            },
            'model': {
                'type': 'MLP',
                'mlp': {
                    'input_dim': 784,
                    'hidden_dims': [32, 16],
                    'output_dim': 10
                }
            },
            'training': {
                'epochs': 1,
                'learning_rate': 0.01,
                'weight_decay': 1e-4,
                'optimizer': 'sgd'
            },
            'env': {
                'device': 'cpu',
                'ray': {
                    'num_cpus': 2,
                    'num_gpus': 0,
                    'object_store_memory': 100000000
                },
                'dataloader': {
                    'num_workers': 0,
                    'pin_memory': False
                }
            },
            'federated': {
                'num_rounds': 2,
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'min_available_clients': 2
            },
            'strategy': {
                'name': 'fedavg'
            },
            'trust': {
                'mode': 'hybrid',
                'threshold': 0.5
            }
        }
    
    def test_partitioning_strategies_smoke(self):
        """Smoke test for all partitioning strategies."""
        try:
            from utils.partitioning import PartitionerRegistry
            
            # Test that all strategies can be created
            strategies = PartitionerRegistry.list_strategies()
            
            for strategy_name in strategies:
                with self.subTest(strategy=strategy_name):
                    try:
                        partitioner = PartitionerRegistry.get_partitioner(strategy_name)
                        self.assertIsNotNone(partitioner)
                    except Exception as e:
                        self.fail(f"Failed to create {strategy_name} partitioner: {e}")
                        
        except ImportError as e:
            self.skipTest(f"Cannot import partitioning module: {e}")
    
    def test_dataset_registry_smoke(self):
        """Smoke test for dataset registry."""
        try:
            from utils.dataset_registry import DatasetRegistry
            
            # Test that all dataset loaders can be created
            datasets = DatasetRegistry.list_datasets()
            
            for dataset_name in datasets:
                with self.subTest(dataset=dataset_name):
                    try:
                        loader = DatasetRegistry.get_loader(dataset_name)
                        self.assertIsNotNone(loader)
                    except Exception as e:
                        self.fail(f"Failed to create {dataset_name} loader: {e}")
                        
        except ImportError as e:
            self.skipTest(f"Cannot import dataset registry: {e}")
    
    def test_model_creation_smoke(self):
        """Smoke test for model creation."""
        try:
            from models.model import MLP, LSTM
            
            # Test MLP creation
            try:
                mlp = MLP(
                    input_dim=784,
                    output_dim=10
                )
                self.assertIsNotNone(mlp)
            except Exception as e:
                self.fail(f"Failed to create MLP: {e}")
            
            # Test LSTM creation
            try:
                lstm = LSTM(
                    input_dim=784,
                    hidden_dim=32,
                    num_layers=2,
                    output_dim=10
                )
                self.assertIsNotNone(lstm)
            except Exception as e:
                self.fail(f"Failed to create LSTM: {e}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import models: {e}")
    
    def test_trust_evaluator_smoke(self):
        """Smoke test for trust evaluator."""
        try:
            from trust_module.trust_evaluator import TrustEvaluator
            
            # Test trust evaluator creation
            try:
                trust_evaluator = TrustEvaluator(trust_mode='hybrid', threshold=0.5)
                self.assertIsNotNone(trust_evaluator)
            except Exception as e:
                self.fail(f"Failed to create trust evaluator: {e}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import trust evaluator: {e}")
    
    def test_metrics_logger_smoke(self):
        """Smoke test for metrics logging."""
        try:
            from utils.metrics_logger import CSVLogger, JSONLogger
            
            # Test CSV logger creation
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                csv_logger = CSVLogger(tmp_path)
                csv_logger.log_scalar('test_metric', 1.0, 1)
                csv_logger.close()
                
                # Check file was created
                self.assertTrue(os.path.exists(tmp_path))
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Test JSON logger creation
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                json_logger = JSONLogger(tmp_path)
                json_logger.log_scalar('test_metric', 1.0, 1)
                json_logger.close()
                
                # Check file was created
                self.assertTrue(os.path.exists(tmp_path))
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except ImportError as e:
            self.skipTest(f"Cannot import metrics logger: {e}")
    
    def test_ray_utils_smoke(self):
        """Smoke test for Ray utilities."""
        try:
            from utils.ray_utils import RayResourceManager
            
            # Test Ray resource manager creation (without actual Ray initialization)
            ray_config = {
                'num_cpus': 2,
                'num_gpus': 0,
                'object_store_memory': 100000000
            }
            
            try:
                ray_manager = RayResourceManager(ray_config)
                self.assertIsNotNone(ray_manager)
                # Don't actually initialize Ray in tests
            except Exception as e:
                self.fail(f"Failed to create Ray resource manager: {e}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import Ray utils: {e}")
    
    def test_configuration_loading_smoke(self):
        """Smoke test for configuration schemas."""
        try:
            from config.schemas import Config, DatasetConfig, ModelConfig
            
            # Test that configuration classes can be created
            try:
                dataset_config = DatasetConfig()
                self.assertIsNotNone(dataset_config)
                
                model_config = ModelConfig()
                self.assertIsNotNone(model_config)
                
                main_config = Config()
                self.assertIsNotNone(main_config)
                
            except Exception as e:
                self.fail(f"Failed to create configuration objects: {e}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import configuration schemas: {e}")


class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported without errors."""
    
    def test_import_utils_modules(self):
        """Test importing utils modules."""
        modules_to_test = [
            'utils.partitioning',
            'utils.dataset_registry',
            'utils.metrics_logger',
            'utils.ray_utils'
        ]
        
        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    # Skip if dependencies are missing
                    if any(dep in str(e) for dep in ['torch', 'ray', 'flwr', 'numpy', 'pandas']):
                        self.skipTest(f"Skipping {module_name} due to missing dependencies: {e}")
                    else:
                        self.fail(f"Failed to import {module_name}: {e}")
    
    def test_import_core_modules(self):
        """Test importing core modules."""
        modules_to_test = [
            'models.model',
            'trust_module.trust_evaluator',
            'config.schemas'
        ]
        
        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    # Skip if dependencies are missing
                    if any(dep in str(e) for dep in ['torch', 'omegaconf', 'numpy']):
                        self.skipTest(f"Skipping {module_name} due to missing dependencies: {e}")
                    else:
                        self.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    unittest.main()
