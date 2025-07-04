#!/usr/bin/env python3
"""
Standalone runner for the refactored TRUST-MCNet system.
This script properly sets up the Python path and runs the experiment.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now we can import all modules with absolute imports
if __name__ == "__main__":
    # Import after path setup
    from train_refactored import main
    
    # Run the main function
    main()
