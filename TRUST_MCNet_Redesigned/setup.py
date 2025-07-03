#!/usr/bin/env python3
"""
Setup script for TRUST-MCNet redesigned framework.

This script helps with installation and initial setup.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is suitable."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying installation...")
    
    packages = [
        "hydra-core",
        "omegaconf", 
        "ray",
        "flwr",
        "torch",
        "pandas",
        "numpy",
        "scipy"
    ]
    
    all_good = True
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - not found")
            all_good = False
    
    return all_good


def create_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("✅ Created data/ directory")
    else:
        print("✅ data/ directory exists")


def main():
    """Main setup function."""
    print("TRUST-MCNet Redesigned - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found")
        print("   Please run this script from the TRUST_MCNet_Redesigned directory")
        sys.exit(1)
    
    # Create data directory
    create_data_directory()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Some packages failed to install correctly")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run a basic example:")
    print("   python train.py federated.num_rounds=2")
    print("\n2. Run example script:")
    print("   python examples.py")
    print("\n3. See README.md for more usage examples")
    print("=" * 50)


if __name__ == "__main__":
    main()
