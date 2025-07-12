"""
Unit tests for dataset registry functionality.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.dataset_registry import DatasetRegistry, MNISTLoader, CSVLoader, DataManager


class TestDatasetRegistry(unittest.TestCase):
    """Test suite for dataset registry."""
    
    def test_registry_functionality(self):
        """Test basic registry functionality."""
        # Test listing datasets
        datasets = DatasetRegistry.list_datasets()
        self.assertIn('mnist', datasets)
        self.assertIn('cifar10', datasets)
        self.assertIn('custom_csv', datasets)
        
        # Test getting loaders
        mnist_loader = DatasetRegistry.get_loader('mnist')
        self.assertIsInstance(mnist_loader, MNISTLoader)
        
        csv_loader = DatasetRegistry.get_loader('custom_csv')
        self.assertIsInstance(csv_loader, CSVLoader)
        
        # Test unknown dataset
        with self.assertRaises(ValueError):
            DatasetRegistry.get_loader('unknown_dataset')
    
    def test_mnist_loader_info(self):
        """Test MNIST loader information methods."""
        loader = MNISTLoader()
        
        # Test data shape
        config = {'transforms': {'normalize': True}}
        shape = loader.get_data_shape(config)
        self.assertEqual(shape, (1, 28, 28))
        
        # Test number of classes
        num_classes = loader.get_num_classes(config)
        self.assertEqual(num_classes, 10)
        
        # Test binary classification
        binary_config = {
            'binary_classification': {'enabled': True}
        }
        num_classes_binary = loader.get_num_classes(binary_config)
        self.assertEqual(num_classes_binary, 2)
    
    def test_csv_loader_with_temp_file(self):
        """Test CSV loader with temporary file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Write test data
            tmp_file.write('feature1,feature2,target\n')
            tmp_file.write('1.0,2.0,0\n')
            tmp_file.write('2.0,3.0,1\n')
            tmp_file.write('3.0,4.0,0\n')
            tmp_file.write('4.0,5.0,1\n')
            tmp_file_path = tmp_file.name
        
        try:
            loader = CSVLoader()
            config = {
                'path': tmp_file_path,
                'csv': {
                    'target_column': 'target',
                    'feature_columns': ['feature1', 'feature2']
                }
            }
            
            # Test loading
            dataset, test_dataset = loader.load(config)
            self.assertIsNotNone(dataset)
            self.assertIsNone(test_dataset)  # CSV loader doesn't provide test set
            
            # Test data shape and classes
            shape = loader.get_data_shape(config)
            self.assertEqual(len(shape), 1)  # Should be 1D feature vector
            
            num_classes = loader.get_num_classes(config)
            self.assertEqual(num_classes, 2)  # Default for CSV
            
        finally:
            # Cleanup temporary file
            os.unlink(tmp_file_path)
    
    def test_data_manager(self):
        """Test DataManager functionality."""
        # Test with MNIST configuration
        mnist_config = {
            'name': 'mnist',
            'path': './test_data',
            'transforms': {'normalize': True}
        }
        
        data_manager = DataManager(mnist_config)
        
        # Test getting data info
        info = data_manager.get_data_info()
        self.assertEqual(info['dataset_name'], 'mnist')
        self.assertEqual(info['data_shape'], (1, 28, 28))
        self.assertEqual(info['num_classes'], 10)
    
    def test_registry_extensibility(self):
        """Test that registry can be extended with custom loaders."""
        from utils.dataset_registry import DatasetLoader
        
        class DummyLoader(DatasetLoader):
            def load(self, config):
                return None, None
            
            def get_data_shape(self, config):
                return (3, 32, 32)
            
            def get_num_classes(self, config):
                return 10
        
        # Register dummy loader
        DatasetRegistry.register_loader('dummy', DummyLoader)
        
        # Test that it's registered
        datasets = DatasetRegistry.list_datasets()
        self.assertIn('dummy', datasets)
        
        # Test that it can be retrieved
        dummy_loader = DatasetRegistry.get_loader('dummy')
        self.assertIsInstance(dummy_loader, DummyLoader)
    
    def test_invalid_csv_file(self):
        """Test CSV loader with invalid file."""
        loader = CSVLoader()
        config = {
            'path': '/nonexistent/file.csv',
            'csv': {
                'target_column': 'target'
            }
        }
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            loader.load(config)


class TestCSVDatasetHandling(unittest.TestCase):
    """Test suite for CSV dataset handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_csv_with_missing_values(self):
        """Test CSV handling with missing values."""
        # Create CSV with missing values
        csv_path = os.path.join(self.temp_dir, 'test_missing.csv')
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, None, 4.0],
            'feature2': [2.0, None, 4.0, 5.0],
            'target': [0, 1, 0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        loader = CSVLoader()
        config = {
            'path': csv_path,
            'csv': {
                'target_column': 'target'
            },
            'preprocessing': {
                'impute_missing': True
            }
        }
        
        # Should load successfully with imputation
        try:
            dataset, _ = loader.load(config)
            self.assertIsNotNone(dataset)
        except Exception as e:
            self.fail(f"CSV loading with missing values failed: {e}")
    
    def test_csv_with_standardization(self):
        """Test CSV with standardization preprocessing."""
        csv_path = os.path.join(self.temp_dir, 'test_standardize.csv')
        data = pd.DataFrame({
            'feature1': [1.0, 100.0, 200.0, 300.0],
            'feature2': [10.0, 20.0, 30.0, 40.0],
            'target': [0, 1, 0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        loader = CSVLoader()
        config = {
            'path': csv_path,
            'csv': {
                'target_column': 'target'
            },
            'preprocessing': {
                'standardize': True
            }
        }
        
        # Should load and standardize successfully
        try:
            dataset, _ = loader.load(config)
            self.assertIsNotNone(dataset)
        except Exception as e:
            self.fail(f"CSV loading with standardization failed: {e}")


if __name__ == '__main__':
    unittest.main()
