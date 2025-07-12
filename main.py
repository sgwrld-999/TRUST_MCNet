#!/usr/bin/env python3
"""
TRUST-MCNet Main Entry Point
============================

This script provides the main entry point for running TRUST-MCNet simulations.
It imports and runs the main simulation from the examples directory.

Usage:
    python main.py [options]
    
For detailed options, see examples/start_simulation.py
"""

import sys
import os
from pathlib import Path
import subprocess

# Add src and examples to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
examples_path = current_dir / "examples"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(examples_path))

def main():
    """Main entry point that runs the simulation with provided arguments."""
    try:
        # Change to project root directory
        os.chdir(current_dir)
        
        # Run the simulation script with all provided arguments
        args = sys.argv[1:]  # Get all command line arguments except script name
        cmd = [sys.executable, str(examples_path / "start_simulation.py")] + args
        
        # Execute the simulation
        result = subprocess.run(cmd, check=True)
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Simulation failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: start_simulation.py not found in examples directory")
        return 1
    except Exception as e:
        print(f"Error: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
