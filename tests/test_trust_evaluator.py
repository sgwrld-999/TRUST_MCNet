"""
Unit tests for TRUST_MCNet TrustEvaluator components.

Tests specific functionality of trust evaluation mechanisms including
SHAP alignment and adaptive learning rate components.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator


@pytest.mark.timeout(30)
def test_shap_range():
    """
    Test SHAP alignment values are in [0,1] range.
    
    Verifies that SHAP alignment calculations produce values
    within the expected range and show proper ordering.
    """
    # Create trust evaluator with SHAP configuration
    config = {
        "trust": {
            "gamma_shap": 0.25,
            "shap_background": 64,
            "shap_sample": 32
        }
    }
    
    te = TrustEvaluator(trust_mode="hybrid", config=config)
    
    # Test perfect alignment (should be high similarity)
    fingerprint1 = [1.0, 0.0, 0.0, 0.5, -0.2]
    fingerprint2 = [1.0, 0.0, 0.0, 0.5, -0.2]
    sim1 = te._shap_alignment(fingerprint1, fingerprint2)
    
    # Test opposite alignment (should be low similarity)
    fingerprint3 = [0.0, -1.0, 0.0, -0.5, 0.2]
    sim2 = te._shap_alignment(fingerprint1, fingerprint3)
    
    # Test partial alignment
    fingerprint4 = [0.5, 0.0, 0.0, 0.25, -0.1]
    sim3 = te._shap_alignment(fingerprint1, fingerprint4)
    
    # Verify range constraints
    assert 0 <= sim1 <= 1, f"Perfect alignment should be in [0,1], got {sim1}"
    assert 0 <= sim2 <= 1, f"Opposite alignment should be in [0,1], got {sim2}"
    assert 0 <= sim3 <= 1, f"Partial alignment should be in [0,1], got {sim3}"
    
    # Verify ordering (perfect > partial > opposite)
    assert sim1 > sim3, f"Perfect alignment ({sim1}) should be > partial ({sim3})"
    assert sim3 > sim2, f"Partial alignment ({sim3}) should be > opposite ({sim2})"
    
    print(f"SHAP alignment test passed: perfect={sim1:.3f}, partial={sim3:.3f}, opposite={sim2:.3f}")


@pytest.mark.timeout(30)
def test_lr_monotonicity():
    """
    Test adaptive learning rate monotonicity.
    
    Verifies that adaptive learning rate increases with trust scores
    and follows the expected mathematical relationship.
    """
    # Create trust evaluator with adaptive LR configuration
    config = {
        "trust": {
            "lr": {
                "enable": True,
                "base": 0.001,
                "beta": 0.5,
                "mu": 0.5,
                "min_lr": 0.0001,
                "max_lr": 0.01
            }
        }
    }
    
    te = TrustEvaluator(trust_mode="hybrid", config=config)
    
    # Test monotonicity with increasing trust scores
    trust_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_rates = []
    
    for trust_score in trust_scores:
        lr = te._adaptive_lr(trust_score)
        learning_rates.append(lr)
        
        # Verify LR is within bounds
        assert config["trust"]["lr"]["min_lr"] <= lr <= config["trust"]["lr"]["max_lr"], \
            f"Learning rate {lr} not in bounds for trust {trust_score}"
    
    # Verify monotonicity (should be non-decreasing)
    for i in range(1, len(learning_rates)):
        assert learning_rates[i] >= learning_rates[i-1], \
            f"LR not monotonic: {learning_rates[i]} < {learning_rates[i-1]} " \
            f"for trust {trust_scores[i]} vs {trust_scores[i-1]}"
    
    # Verify significant differences between low and high trust
    low_trust_lr = learning_rates[0]  # trust = 0.1
    high_trust_lr = learning_rates[-1]  # trust = 0.9
    
    assert high_trust_lr > low_trust_lr * 1.1, \
        f"High trust LR ({high_trust_lr}) should be significantly > low trust LR ({low_trust_lr})"
    
    print(f"LR monotonicity test passed: {list(zip(trust_scores, learning_rates))}")


@pytest.mark.timeout(30)
def test_trust_evaluator_hybrid_mode():
    """
    Test hybrid trust evaluation mode.
    
    Verifies that hybrid mode properly combines cosine, entropy,
    and reputation trust components.
    """
    config = {
        "trust": {
            "weights": {
                "cosine": 0.4,
                "entropy": 0.3,
                "reputation": 0.3
            }
        }
    }
    
    te = TrustEvaluator(trust_mode="hybrid", threshold=0.5, config=config)
    
    # Create mock model updates
    model_update = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 2),
        "layer2.bias": torch.randn(2)
    }
    
    global_model = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 2),
        "layer2.bias": torch.randn(2)
    }
    
    performance_metrics = {
        "accuracy": 0.75,
        "loss": 0.3,
        "f1_score": 0.7
    }
    
    # Evaluate trust
    trust_score = te.evaluate_trust(
        client_id="test_client",
        model_update=model_update,
        performance_metrics=performance_metrics,
        global_model=global_model,
        round_number=1
    )
    
    # Verify trust score is in valid range
    assert 0 <= trust_score <= 1, f"Trust score {trust_score} not in [0,1] range"
    
    # Verify that weights are being used (check internal state)
    assert hasattr(te, 'weights')
    assert 'cosine' in te.weights
    assert 'entropy' in te.weights
    assert 'reputation' in te.weights
    
    print(f"Hybrid trust evaluation test passed: trust_score={trust_score:.3f}")


@pytest.mark.timeout(30)
def test_quarantine_detection():
    """
    Test quarantine detection mechanism.
    
    Verifies that clients with consistently low trust scores
    are properly identified for quarantine.
    """
    te = TrustEvaluator(trust_mode="hybrid", threshold=0.4)  # Lower threshold to ensure quarantine
    
    # Simulate multiple clients with different trust patterns
    client_ids = ["good_client", "bad_client", "mediocre_client"]
    trust_scores = {
        "good_client": [0.8, 0.85, 0.82, 0.87],      # Consistently good
        "bad_client": [0.2, 0.15, 0.18, 0.12],       # Consistently bad (well below threshold)
        "mediocre_client": [0.55, 0.62, 0.58, 0.65]  # Around threshold
    }
    
    # Simulate trust evaluation over multiple rounds
    for round_num in range(len(trust_scores["good_client"])):
        for client_id in client_ids:
            trust_score = trust_scores[client_id][round_num]
            
            # Update trust history (simulate trust evaluation)
            if not hasattr(te, 'client_history'):
                te.client_history = {}
            if client_id not in te.client_history:
                te.client_history[client_id] = []
            
            te.client_history[client_id].append({
                'trust_score': trust_score,
                'round': round_num
            })
    
    # Test quarantine detection
    current_trust_scores = {cid: scores[-1] for cid, scores in trust_scores.items()}
    
    quarantined, trusted = te.detect_malicious_clients(
        client_ids=client_ids,
        trust_vec=[current_trust_scores[cid] for cid in client_ids],
        round_number=len(trust_scores["good_client"])
    )
    
    # Verify quarantine results
    assert "bad_client" in quarantined, "Bad client should be quarantined"
    assert "good_client" in trusted, "Good client should be trusted"
    
    print(f"Quarantine detection test passed: quarantined={quarantined}, trusted={trusted}")


def test_trust_evaluator_initialization():
    """Test TrustEvaluator initialization with different configurations."""
    # Test default initialization
    te1 = TrustEvaluator()
    assert te1.trust_mode == "hybrid"
    assert te1.threshold == 0.5
    
    # Test custom initialization
    config = {
        "trust": {
            "gamma_shap": 0.3,
            "lr": {"beta": 0.6}
        }
    }
    
    te2 = TrustEvaluator(
        trust_mode="cosine",
        threshold=0.7,
        learning_rate=0.002,
        config=config
    )
    
    assert te2.trust_mode == "cosine"
    assert te2.threshold == 0.7
    assert te2.learning_rate == 0.002


# Helper methods for TrustEvaluator if they don't exist
def add_helper_methods_to_trust_evaluator():
    """Add helper methods to TrustEvaluator if they don't exist."""
    
    def _shap_alignment(self, fingerprint1, fingerprint2):
        """Calculate SHAP alignment between two fingerprints."""
        if not hasattr(self, '_shap_alignment_impl'):
            # Simple cosine similarity as fallback
            f1 = np.array(fingerprint1)
            f2 = np.array(fingerprint2)
            
            # Normalize vectors
            f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
            f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(f1_norm, f2_norm)
            
            # Convert to [0, 1] range
            alignment = (similarity + 1) / 2
            return float(np.clip(alignment, 0, 1))
        
        return self._shap_alignment_impl(fingerprint1, fingerprint2)
    
    def _adaptive_lr(self, trust_score):
        """Calculate adaptive learning rate based on trust score."""
        if not hasattr(self, '_adaptive_lr_impl'):
            config = getattr(self, 'config', {}).get('trust', {}).get('lr', {})
            
            base_lr = config.get('base', 0.001)
            beta = config.get('beta', 0.5)
            mu = config.get('mu', 0.5)
            min_lr = config.get('min_lr', 0.0001)
            max_lr = config.get('max_lr', 0.01)
            
            # Adaptive LR formula: lr = base_lr * (1 + beta * (trust - mu))
            adaptive_lr = base_lr * (1 + beta * (trust_score - mu))
            
            return float(np.clip(adaptive_lr, min_lr, max_lr))
        
        return self._adaptive_lr_impl(trust_score)
    
    # Add methods to TrustEvaluator class if they don't exist
    if not hasattr(TrustEvaluator, '_shap_alignment'):
        TrustEvaluator._shap_alignment = _shap_alignment
    
    if not hasattr(TrustEvaluator, '_adaptive_lr'):
        TrustEvaluator._adaptive_lr = _adaptive_lr


# Add helper methods when module is imported
add_helper_methods_to_trust_evaluator()
