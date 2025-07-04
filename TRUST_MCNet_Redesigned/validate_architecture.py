"""
Simple validation script for the refactored TRUST-MCNet architecture.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_module_structure():
    """Test that all modules exist and have the expected structure."""
    print("Testing module structure...")
    
    # Check core modules
    core_modules = ['interfaces.py', 'abstractions.py', 'exceptions.py', 'types.py']
    core_dir = current_dir / 'core'
    
    if not core_dir.exists():
        print(f"❌ Core directory not found: {core_dir}")
        return False
    
    for module in core_modules:
        module_path = core_dir / module
        if not module_path.exists():
            print(f"❌ Core module not found: {module_path}")
            return False
        else:
            print(f"✅ Found core module: {module}")
    
    # Check component directories
    component_dirs = ['data', 'models_new', 'partitioning', 'strategies', 
                     'trust_new', 'metrics', 'clients_new', 'experiments']
    
    for comp_dir in component_dirs:
        dir_path = current_dir / comp_dir
        if not dir_path.exists():
            print(f"❌ Component directory not found: {dir_path}")
            return False
        
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            print(f"❌ Component __init__.py not found: {init_file}")
            return False
        
        print(f"✅ Found component directory: {comp_dir}")
    
    return True

def test_basic_imports():
    """Test basic imports without executing complex logic."""
    print("\nTesting basic imports...")
    
    try:
        # Test core imports
        sys.path.insert(0, str(current_dir / 'core'))
        import types as core_types
        print("✅ Core types module imported")
        
        import exceptions as core_exceptions
        print("✅ Core exceptions module imported")
        
        # Reset path
        sys.path = sys.path[1:]
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_content():
    """Test that files have expected content."""
    print("\nTesting file content...")
    
    # Check if core files have expected classes/functions
    interfaces_file = current_dir / 'core' / 'interfaces.py'
    if interfaces_file.exists():
        content = interfaces_file.read_text()
        if 'DataLoaderInterface' in content and 'ModelInterface' in content:
            print("✅ Interfaces file has expected interfaces")
        else:
            print("❌ Interfaces file missing expected interfaces")
            return False
    
    # Check experiments file
    experiments_file = current_dir / 'experiments' / '__init__.py'
    if experiments_file.exists():
        content = experiments_file.read_text()
        if 'FederatedExperiment' in content and 'ExperimentRegistry' in content:
            print("✅ Experiments file has expected classes")
        else:
            print("❌ Experiments file missing expected classes")
            return False
    
    return True

def main():
    """Run all validation tests."""
    print("🔍 Validating TRUST-MCNet Refactored Architecture")
    print("=" * 60)
    
    tests = [
        test_module_structure,
        test_basic_imports,
        test_file_content
    ]
    
    all_passed = True
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Architecture validation completed successfully!")
        print("✅ All modules are properly structured")
        print("✅ Basic imports work correctly")
        print("✅ Files contain expected content")
        print("\nThe refactored TRUST-MCNet architecture is ready for use.")
        print("\nNext steps:")
        print("1. Install required dependencies (torch, numpy, flwr, etc.)")
        print("2. Run: python train_refactored.py")
        print("3. Or use: python train_refactored.py dataset.num_clients=100")
    else:
        print("❌ Architecture validation failed!")
        print("Please check the errors above and fix the issues.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
