#!/usr/bin/env python3
"""
Integration Test Script for TRUST_MCNet Flower Integration

This script tests the key components of the integrated federated learning system.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
    try:
        from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
        print("✓ TrustEvaluator import successful")
        
        # Test TrustEvaluator instantiation
        evaluator = TrustEvaluator()
        print("✓ TrustEvaluator instantiation successful")
        
        # Test TrustWeightedStrategy import with fallback
        try:
            from trust_mcnet.strategies.trust_weighted_strategy import TrustWeightedStrategy
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'trust_weighted_strategy', 
                str(src_path / 'trust_mcnet/strategies/trust_weighted_strategy.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TrustWeightedStrategy = module.TrustWeightedStrategy
        
        print("✓ TrustWeightedStrategy import successful")
        
        # Test strategy instantiation
        strategy = TrustWeightedStrategy(trust_evaluator=evaluator)
        print("✓ TrustWeightedStrategy instantiation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_main_modes():
    """Test that main.py modes work correctly."""
    print("\nTesting main.py modes...")
    
    # Test help
    try:
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Main help successful")
        else:
            print(f"✗ Main help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Main help test failed: {e}")
        return False
    
    # Test simulation help
    try:
        result = subprocess.run([sys.executable, "main.py", "simulation", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "TRUST_MCNet IoT" in result.stdout:
            print("✓ Simulation help successful")
        else:
            print(f"✗ Simulation help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Simulation help test failed: {e}")
        return False
    
    # Test flower server help
    try:
        result = subprocess.run([sys.executable, "main.py", "flower_server", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Flower server help successful")
        else:
            print(f"✗ Flower server help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Flower server help test failed: {e}")
        return False
    
    return True

def test_config_files():
    """Test that configuration files are properly formatted."""
    print("\nTesting configuration files...")
    
    try:
        import yaml
        
        # Test main config file
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for federated section
        if 'federated' in config:
            print("✓ Federated configuration found")
        else:
            print("✗ Federated configuration missing")
            return False
        
        # Check for simulation section
        if 'simulation' in config:
            print("✓ Simulation configuration found")
        else:
            print("✗ Simulation configuration missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("TRUST_MCNet Flower Integration Test")
    print("=" * 40)
    
    os.chdir(current_dir)
    
    tests = [
        test_imports,
        test_main_modes,
        test_config_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration successful.")
        return 0
    else:
        print("❌ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
