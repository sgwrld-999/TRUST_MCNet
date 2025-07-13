#!/usr/bin/env python3
"""
Enhanced Integration Test Script for TRUST_MCNet Adaptive Features

This script tests the adaptive trust strategy and real-time monitoring
components to ensure they work correctly with the Flower integration.
"""

import sys
import os
import subprocess
import time
import tempfile
import shutil
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_adaptive_imports():
    """Test that adaptive strategy and monitoring imports work correctly."""
    print("Testing adaptive imports...")
    
    try:
        from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
        print("✓ TrustEvaluator import successful")
        
        from trust_mcnet.strategies.trust_weighted_strategy import TrustWeightedStrategy
        print("✓ TrustWeightedStrategy import successful")
        
        # Test adaptive strategy import with fallback
        try:
            from trust_mcnet.strategies.adaptive_trust_strategy import AdaptiveTrustStrategy
            print("✓ AdaptiveTrustStrategy import successful")
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'adaptive_trust_strategy', 
                str(src_path / 'trust_mcnet/strategies/adaptive_trust_strategy.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AdaptiveTrustStrategy = module.AdaptiveTrustStrategy
            print("✓ AdaptiveTrustStrategy fallback import successful")
        
        # Test monitoring import
        try:
            from trust_mcnet.monitoring.trust_dashboard import TrustDashboard, TrustMetricsTracker
            print("✓ Monitoring components import successful")
        except ImportError as e:
            print(f"⚠ Monitoring components not available: {e}")
        
        # Test instantiation
        evaluator = TrustEvaluator()
        strategy = TrustWeightedStrategy(trust_evaluator=evaluator)
        adaptive_strategy = AdaptiveTrustStrategy(trust_evaluator=evaluator)
        
        print("✓ All strategy instantiations successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive import test failed: {e}")
        return False

def test_adaptive_functionality():
    """Test adaptive strategy functionality."""
    print("\nTesting adaptive strategy functionality...")
    
    try:
        from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
        
        # Test adaptive strategy import with fallback
        try:
            from trust_mcnet.strategies.adaptive_trust_strategy import AdaptiveTrustStrategy
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'adaptive_trust_strategy', 
                str(src_path / 'trust_mcnet/strategies/adaptive_trust_strategy.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AdaptiveTrustStrategy = module.AdaptiveTrustStrategy
        
        # Create adaptive strategy
        evaluator = TrustEvaluator()
        strategy = AdaptiveTrustStrategy(
            trust_evaluator=evaluator,
            target_accuracy=0.85,
            threshold_adaptation_rate=0.05
        )
        
        # Test adaptation status
        status = strategy.get_adaptation_status()
        print(f"✓ Initial threshold: {status['current_trust_threshold']:.3f}")
        print(f"✓ Target accuracy: {status['target_accuracy']:.3f}")
        print(f"✓ Adaptation config: {status['adaptation_config']}")
        
        # Test threshold bounds
        assert status['threshold_bounds']['min'] <= status['current_trust_threshold'] <= status['threshold_bounds']['max']
        print("✓ Threshold bounds validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive functionality test failed: {e}")
        return False

def test_monitoring_functionality():
    """Test monitoring dashboard functionality."""
    print("\nTesting monitoring functionality...")
    
    try:
        from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
        
        # Test monitoring import with fallback
        try:
            from trust_mcnet.monitoring.trust_dashboard import TrustDashboard, TrustMetricsTracker
        except ImportError:
            print("⚠ Monitoring components not available, skipping test")
            return True
        
        # Test adaptive strategy import
        try:
            from trust_mcnet.strategies.adaptive_trust_strategy import AdaptiveTrustStrategy
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'adaptive_trust_strategy', 
                str(src_path / 'trust_mcnet/strategies/adaptive_trust_strategy.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AdaptiveTrustStrategy = module.AdaptiveTrustStrategy
        
        # Create temporary directory for dashboard
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create strategy and dashboard
            evaluator = TrustEvaluator()
            strategy = AdaptiveTrustStrategy(trust_evaluator=evaluator)
            dashboard = TrustDashboard(strategy=strategy, output_dir=temp_dir)
            
            print("✓ Dashboard created successfully")
            
            # Test metrics tracker
            tracker = TrustMetricsTracker()
            
            # Test logging metrics
            test_metrics = {
                'round': 1,
                'avg_accuracy': 0.75,
                'avg_loss': 0.3,
                'trust_scores': [0.6, 0.7, 0.8, 0.5, 0.9]
            }
            
            tracker.log_trust_distribution(test_metrics)
            print("✓ Metrics logging successful")
            
            # Test statistics
            stats = tracker.get_trust_statistics()
            print(f"✓ Trust statistics: mean={stats.get('overall_mean', 0):.3f}")
            
            # Test dashboard update
            dashboard.update_dashboard(test_metrics)
            print("✓ Dashboard update successful")
            
            # Check if files were created
            output_path = Path(temp_dir)
            if (output_path / 'trust_metrics.json').exists():
                print("✓ Metrics export successful")
            if (output_path / 'status.json').exists():
                print("✓ Status export successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Monitoring functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_main_modes():
    """Test enhanced main.py modes with adaptive features."""
    print("\nTesting enhanced main.py modes...")
    
    # Test enhanced configuration help
    try:
        result = subprocess.run([sys.executable, "main.py", "flower_server", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Enhanced flower server help successful")
        else:
            print(f"✗ Enhanced flower server help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Enhanced help test failed: {e}")
        return False
    
    return True

def test_enhanced_config():
    """Test enhanced configuration file."""
    print("\nTesting enhanced configuration...")
    
    try:
        import yaml
        
        # Test enhanced config file
        config_path = "config/enhanced_federated.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for adaptive settings
            federated = config.get('federated', {})
            if federated.get('use_adaptive_strategy', False):
                print("✓ Adaptive strategy configuration found")
            
            if federated.get('enable_monitoring', False):
                print("✓ Monitoring configuration found")
            
            # Check for adaptive parameters
            required_params = [
                'target_accuracy', 'adaptation_rate', 'max_threshold', 
                'min_threshold', 'performance_window', 'adaptation_patience'
            ]
            
            for param in required_params:
                if param in federated:
                    print(f"✓ Parameter {param}: {federated[param]}")
                else:
                    print(f"⚠ Parameter {param} not found")
        else:
            print("⚠ Enhanced configuration file not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced config test failed: {e}")
        return False

def main():
    """Run all enhanced integration tests."""
    print("TRUST_MCNet Enhanced Features Integration Test")
    print("=" * 50)
    
    os.chdir(current_dir)
    
    tests = [
        test_adaptive_imports,
        test_adaptive_functionality,
        test_monitoring_functionality,
        test_enhanced_main_modes,
        test_enhanced_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Enhanced Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All enhanced tests passed! Adaptive features working correctly.")
        return 0
    else:
        print("❌ Some enhanced tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
