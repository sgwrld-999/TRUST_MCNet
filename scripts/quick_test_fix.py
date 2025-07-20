#!/usr/bin/env python3
"""
Quick test to validate the accuracy fix for TRUST_MCNet.
Run this after applying the configuration changes.
"""

import subprocess
import time
import json
from pathlib import Path

def run_test():
    """Run a quick test with the fixed configuration."""
    
    print("=== TRUST_MCNet Accuracy Fix Validation ===\n")
    
    # Test with fixed configuration
    print("1. Testing with FIXED configuration...")
    print("   Running: python enhanced_simulation.py dataset=iot_fixed federated.num_rounds=3")
    
    try:
        # Run simulation with fixed config
        result = subprocess.run([
            "python", "enhanced_simulation.py", 
            "dataset=iot_fixed", 
            "federated.num_rounds=3",
            "simulation.enable_trust_evaluation=true"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Simulation completed successfully!")
            
            # Look for results in output
            output_lines = result.stdout.split('\n')
            
            # Extract key metrics
            for line in output_lines:
                if 'average accuracy' in line.lower() or 'global accuracy' in line.lower():
                    print(f"   {line.strip()}")
                elif 'trust' in line.lower() and 'client' in line.lower():
                    print(f"   {line.strip()}")
            
        else:
            print("❌ Simulation failed!")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Simulation timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. If global accuracy improved significantly (>40%), the fix worked!")
    print("2. If still low, run the diagnostic script:")
    print("   python scripts/diagnose_accuracy_issue.py") 
    print("3. For detailed analysis, check the results/ directory")

if __name__ == "__main__":
    run_test()
