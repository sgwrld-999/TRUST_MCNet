"""
Test suite runner for TRUST-MCNet.
"""

import unittest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all test suites."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test_suite(test_module):
    """Run a specific test suite."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TRUST-MCNet tests')
    parser.add_argument('--module', '-m', help='Specific test module to run')
    parser.add_argument('--list', '-l', action='store_true', help='List available test modules')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test modules:")
        test_files = list(Path(__file__).parent.glob('test_*.py'))
        for test_file in test_files:
            module_name = test_file.stem
            print(f"  {module_name}")
        sys.exit(0)
    
    if args.module:
        success = run_specific_test_suite(args.module)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
