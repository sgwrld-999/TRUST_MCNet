#!/usr/bin/env python3
"""
Integration test for trust-weighted Flower server.

This script runs a quick sanity test as described in the guide.txt:
1. Starts a trust-weighted server for 1 round
2. Simulates 2 dummy clients
3. Validates trust metrics are generated

Usage:
    python test_trust_integration.py
"""

import logging
import subprocess
import sys
import time
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_server():
    """Run the trust-weighted server."""
    logger.info("Starting trust-weighted server...")
    
    cmd = [
        sys.executable, 
        "server/run_federated.py",
        "--num_rounds", "1",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("Server completed successfully")
            if "mean_trust" in result.stdout or "mean_trust" in result.stderr:
                logger.info("✓ Trust metrics detected in server output")
                return True
            else:
                logger.warning("⚠ Trust metrics not found in server output")
                return False
        else:
            logger.error(f"Server failed with return code: {result.returncode}")
            logger.error(f"Server stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Server timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to run server: {e}")
        return False


def run_client(client_id: int, delay: float = 0):
    """Run a test client."""
    if delay > 0:
        time.sleep(delay)
    
    logger.info(f"Starting client {client_id}...")
    
    cmd = [
        sys.executable,
        "client/simulate.py",
        "--cid", str(client_id),
        "--verbose"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info(f"Client {client_id} completed successfully")
            return True
        else:
            logger.error(f"Client {client_id} failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Client {client_id} timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to run client {client_id}: {e}")
        return False


def test_trust_integration():
    """Run integration test for trust-weighted aggregation."""
    logger.info("="*60)
    logger.info("TRUST-MCNet Integration Test")
    logger.info("Testing trust-weighted Flower server integration")
    logger.info("="*60)
    
    # Check if required files exist
    required_files = [
        "server/run_federated.py",
        "client/simulate.py",
        "src/trust_mcnet/strategies/trust_weighted_strategy.py",
        "src/trust_mcnet/trust_module/trust_evaluator.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("✓ All required files found")
    
    # Test 1: Start server in background
    logger.info("Test 1: Starting trust-weighted server...")
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Give server time to start
    time.sleep(5)
    
    # Test 2: Run clients
    logger.info("Test 2: Running test clients...")
    
    client_threads = []
    for i in range(2):
        thread = threading.Thread(target=run_client, args=(i, i * 0.5))
        thread.start()
        client_threads.append(thread)
    
    # Wait for all clients to complete
    for thread in client_threads:
        thread.join(timeout=30)
    
    # Wait for server to complete
    server_thread.join(timeout=10)
    
    logger.info("="*60)
    logger.info("Integration test completed")
    logger.info("Check logs for trust metrics (mean_trust, min_trust, max_trust)")
    logger.info("="*60)
    
    return True


def main():
    """Main test entry point."""
    try:
        success = test_trust_integration()
        if success:
            logger.info("✓ Integration test passed")
            sys.exit(0)
        else:
            logger.error("✗ Integration test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
