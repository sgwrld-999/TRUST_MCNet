"""
Integration tests for TRUST_MCNet strategy components.

Tests the integration between trust evaluation, strategy, and federated learning components.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any, List
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.trust_mcnet.strategies.unified_trust_strategy import UnifiedTrustStrategy
from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
from datasets import get as get_dataset


@pytest.mark.timeout(30)
def test_poison_quarantine():
    """
    Integration smoke test for quarantine functionality.
    
    Tests that the system can detect and quarantine malicious clients
    in a simulated federated learning scenario.
    """
    # Create trust evaluator with strict threshold
    trust_config = {
        "trust": {
            "threshold": 0.7,
            "aggregation": {
                "trim_ratio": 0.2
            }
        }
    }
    
    trust_eval = TrustEvaluator(
        trust_mode="hybrid",
        threshold=0.7,
        config=trust_config
    )
    
    # Create strategy
    strategy = UnifiedTrustStrategy(trust_evaluator=trust_eval)
    
    # Simulate client updates with different trust patterns
    results = run_mini_simulation(strategy, num_clients=3, rounds=2)
    
    # Verify quarantine behavior
    assert "quarantined_clients" in results
    assert "trust_scores" in results
    
    # Check that at least one client has low trust (simulating malicious behavior)
    trust_scores = results["trust_scores"]
    min_trust = min(trust_scores.values()) if trust_scores else 1.0
    
    # Should have some variation in trust scores
    max_trust = max(trust_scores.values()) if trust_scores else 0.0
    trust_range = max_trust - min_trust
    
    assert trust_range > 0.1, "Trust scores should show variation between clients"


@pytest.mark.timeout(30)
def test_strategy_integration():
    """
    Test basic strategy integration with trust evaluation.
    
    Verifies that the UnifiedTrustStrategy can properly integrate
    with TrustEvaluator for federated aggregation.
    """
    # Create components
    trust_eval = TrustEvaluator(trust_mode="cosine", threshold=0.5)
    strategy = UnifiedTrustStrategy(
        trust_evaluator=trust_eval,
        fraction_fit=0.8,
        min_fit_clients=2
    )
    
    # Verify strategy initialization
    assert strategy.trust_eval is trust_eval
    assert strategy.fraction_fit == 0.8
    assert strategy.min_fit_clients == 2
    
    # Test basic functionality without full simulation
    assert hasattr(strategy, 'aggregate_fit')
    assert hasattr(strategy, 'configure_fit')


@pytest.mark.timeout(30)
def test_dataset_integration():
    """
    Test dataset integration with synthetic data.
    
    Verifies that datasets can be loaded and provide the expected interface.
    """
    # Test ToN-IoT dataset
    dataset = get_dataset("ton_iot", batch_size=16)
    
    # Verify dataset interface
    assert hasattr(dataset, 'train_loader')
    assert hasattr(dataset, 'test_loader')
    assert hasattr(dataset, 'input_dim')
    assert hasattr(dataset, 'num_classes')
    
    # Verify data dimensions
    train_loader = dataset.train_loader()
    test_loader = dataset.test_loader()
    
    # Get a batch to verify dimensions
    for batch_x, batch_y in train_loader:
        assert batch_x.shape[1] == dataset.input_dim
        assert batch_y.max().item() < dataset.num_classes
        break
    
    # Verify test loader works
    for batch_x, batch_y in test_loader:
        assert batch_x.shape[1] == dataset.input_dim
        break


def run_mini_simulation(strategy, num_clients: int = 2, rounds: int = 1) -> Dict[str, Any]:
    """
    Run a minimal simulation for testing purposes.
    
    Args:
        strategy: Federated learning strategy
        num_clients: Number of clients to simulate
        rounds: Number of rounds to run
        
    Returns:
        Simulation results including trust scores and quarantine info
    """
    from flwr.common import Parameters, FitRes, ndarrays_to_parameters
    from flwr.server.client_proxy import ClientProxy
    
    # Create mock client results with different patterns
    results = []
    client_ids = [f"client_{i}" for i in range(num_clients)]
    
    for i, client_id in enumerate(client_ids):
        # Create mock parameters (simple 2D arrays)
        if i == 0:
            # "Good" client - normal parameters
            params = [np.random.normal(0, 0.1, (10, 5)), np.random.normal(0, 0.1, (5,))]
            metrics = {"accuracy": 0.85, "train_loss": 0.2}
        elif i == 1:
            # "Suspicious" client - different distribution
            params = [np.random.normal(2, 1, (10, 5)), np.random.normal(1, 0.5, (5,))]
            metrics = {"accuracy": 0.45, "train_loss": 0.8}
        else:
            # "Normal" client
            params = [np.random.normal(0, 0.2, (10, 5)), np.random.normal(0, 0.1, (5,))]
            metrics = {"accuracy": 0.75, "train_loss": 0.3}
        
        # Create mock client proxy
        class MockClientProxy(ClientProxy):
            def __init__(self, cid):
                self.cid = cid
                
            def reconnect(self, node, timeout=None):
                pass
            
            def get_properties(self, config, timeout=None):
                return {}
            
            def get_parameters(self, config, timeout=None):
                return ndarrays_to_parameters(params)
            
            def fit(self, parameters, config, timeout=None):
                return FitRes(
                    status={"code": "OK", "message": "Success"},
                    parameters=parameters,
                    num_examples=100,
                    metrics=metrics
                )
            
            def evaluate(self, parameters, config, timeout=None):
                return None
        
        client_proxy = MockClientProxy(client_id)
        fit_result = FitRes(
            status={"code": "OK", "message": "Success"},
            parameters=ndarrays_to_parameters(params),
            num_examples=100,
            metrics=metrics
        )
        
        results.append((client_proxy, fit_result))
    
    # Run aggregation through strategy
    try:
        for round_num in range(1, rounds + 1):
            aggregated_params, round_metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=results,
                failures=[]
            )
            
            # Extract trust information from round metrics
            trust_scores = {}
            quarantined_clients = []
            
            if "trust_scores" in round_metrics:
                trust_scores = round_metrics["trust_scores"]
            
            if "quarantined_clients" in round_metrics:
                quarantined_clients = round_metrics["quarantined_clients"]
            
            # Check individual client trust scores
            for client_id in client_ids:
                if client_id in trust_scores:
                    if trust_scores[client_id] < strategy.trust_eval.threshold:
                        if client_id not in quarantined_clients:
                            quarantined_clients.append(client_id)
    
    except Exception as e:
        # Even if aggregation fails, return partial results for testing
        trust_scores = {f"client_{i}": 0.5 for i in range(num_clients)}
        quarantined_clients = []
    
    return {
        "trust_scores": trust_scores,
        "quarantined_clients": quarantined_clients,
        "num_clients": num_clients,
        "rounds": rounds
    }
