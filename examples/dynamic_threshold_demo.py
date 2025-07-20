#!/usr/bin/env python3
"""
Dynamic Threshold Demonstration for TRUST_MCNet

This script demonstrates the new dynamic trust threshold mechanism
that adaptively calculates trust thresholds based on current client
trust distributions and historical performance.

Key Features:
- Replaces static threshold (0.5) with dynamic calculation
- Adapts based on round number, trust distribution, and performance
- Ensures minimum number of trusted clients
- Maintains trust evaluation algorithm integrity
- Provides comprehensive logging and statistics

Usage:
    python examples/dynamic_threshold_demo.py
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_trust_scores(num_clients: int, round_number: int, scenario: str = "normal") -> Dict[str, float]:
    """
    Simulate trust scores for different scenarios.
    
    Args:
        num_clients: Number of clients to simulate
        round_number: Current federated learning round
        scenario: Simulation scenario
        
    Returns:
        Dictionary of client trust scores
    """
    np.random.seed(42 + round_number)  # Reproducible but varying
    
    if scenario == "normal":
        # Normal distribution around 0.6
        scores = np.random.normal(0.6, 0.15, num_clients)
    elif scenario == "polarized":
        # Mix of high and low trust clients
        high_trust = np.random.normal(0.8, 0.05, num_clients // 2)
        low_trust = np.random.normal(0.3, 0.05, num_clients - num_clients // 2)
        scores = np.concatenate([high_trust, low_trust])
    elif scenario == "low_trust":
        # Generally low trust scores
        scores = np.random.normal(0.3, 0.1, num_clients)
    elif scenario == "high_trust":
        # Generally high trust scores
        scores = np.random.normal(0.8, 0.1, num_clients)
    elif scenario == "declining":
        # Trust declining over rounds
        base_score = max(0.2, 0.8 - (round_number * 0.05))
        scores = np.random.normal(base_score, 0.1, num_clients)
    else:
        # Random scenario
        scores = np.random.uniform(0.1, 0.9, num_clients)
    
    # Clip to valid range
    scores = np.clip(scores, 0.0, 1.0)
    
    # Create client dictionary
    trust_scores = {f"client_{i:02d}": float(score) for i, score in enumerate(scores)}
    
    return trust_scores

def simulate_performance_trend(round_number: int, scenario: str = "improving") -> float:
    """Simulate global model performance over rounds."""
    base_accuracy = 0.3  # Starting accuracy
    
    if scenario == "improving":
        # Gradual improvement with some noise
        trend_accuracy = base_accuracy + (round_number * 0.04)
        noise = np.random.normal(0, 0.02)
        return min(0.95, max(0.1, trend_accuracy + noise))
    elif scenario == "declining":
        # Performance getting worse
        trend_accuracy = base_accuracy + (5 * 0.04) - (max(0, round_number - 5) * 0.03)
        noise = np.random.normal(0, 0.02)
        return min(0.95, max(0.1, trend_accuracy + noise))
    elif scenario == "stable":
        # Stable around 70%
        noise = np.random.normal(0, 0.01)
        return min(0.95, max(0.1, 0.7 + noise))
    else:
        # Random performance
        return np.random.uniform(0.2, 0.9)

def demonstrate_static_vs_dynamic():
    """Demonstrate the difference between static and dynamic thresholds."""
    print("\n" + "="*80)
    print("STATIC vs DYNAMIC THRESHOLD COMPARISON")
    print("="*80)
    
    # Configuration for trust evaluator
    config = {
        'trust': {
            'mode': 'hybrid',
            'threshold': 0.8,  # Static threshold
            'dynamic_threshold': {
                'enabled': True,
                'min_trust_threshold': 0.1,
                'max_trust_threshold': 0.9,
                'min_trusted_clients': 2,
                'target_trusted_ratio': 0.6,
                'threshold_percentile_weight': 0.4,
                'threshold_statistical_weight': 0.3,
                'threshold_adaptive_weight': 0.3
            },
            'weights': {'cosine': 0.4, 'entropy': 0.3, 'reputation': 0.3}
        }
    }
    
    # Initialize trust evaluator
    evaluator = TrustEvaluator(config)
    
    scenarios = ['normal', 'polarized', 'low_trust', 'high_trust']
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario.upper()} ---")
        
        # Simulate 5 rounds
        for round_num in range(1, 6):
            # Generate trust scores
            trust_scores = simulate_trust_scores(8, round_num, scenario)
            global_accuracy = simulate_performance_trend(round_num, "improving")
            
            # Static threshold analysis
            static_threshold = 0.8
            static_trusted = {k: v for k, v in trust_scores.items() if v >= static_threshold}
            
            # Dynamic threshold analysis
            dynamic_trusted, dynamic_threshold = evaluator.get_trusted_clients_dynamic(
                trust_scores, round_num, global_accuracy
            )
            
            # Results comparison
            trust_values = list(trust_scores.values())
            print(f"Round {round_num}:")
            print(f"  Trust scores: mean={np.mean(trust_values):.3f}, "
                  f"std={np.std(trust_values):.3f}, range=[{np.min(trust_values):.3f}, {np.max(trust_values):.3f}]")
            print(f"  Static (0.8):  {len(static_trusted)}/{len(trust_scores)} trusted")
            print(f"  Dynamic ({dynamic_threshold:.3f}): {len(dynamic_trusted)}/{len(trust_scores)} trusted")
            
            if len(static_trusted) == 0:
                print(f"  ❌ Static threshold failed: NO TRUSTED CLIENTS!")
            if len(dynamic_trusted) >= 2:
                print(f"  ✅ Dynamic threshold success: Adequate trusted clients")
            print()

def demonstrate_adaptation_over_time():
    """Demonstrate how dynamic threshold adapts over multiple rounds."""
    print("\n" + "="*80)
    print("DYNAMIC THRESHOLD ADAPTATION OVER TIME")
    print("="*80)
    
    config = {
        'trust': {
            'mode': 'hybrid',
            'dynamic_threshold': {
                'enabled': True,
                'min_trust_threshold': 0.1,
                'max_trust_threshold': 0.9,
                'min_trusted_clients': 2,
                'target_trusted_ratio': 0.6
            },
            'weights': {'cosine': 0.4, 'entropy': 0.3, 'reputation': 0.3}
        }
    }
    
    evaluator = TrustEvaluator(config)
    
    print("Simulating 15 rounds with declining trust scenario...")
    print("Round | Threshold | Trusted | Trust Range    | Performance | Adaptation")
    print("------|-----------|---------|----------------|-------------|------------")
    
    thresholds = []
    trusted_ratios = []
    
    for round_num in range(1, 16):
        # Simulate declining trust scenario
        trust_scores = simulate_trust_scores(10, round_num, "declining")
        global_accuracy = simulate_performance_trend(round_num, "declining")
        
        # Get dynamic trusted clients
        dynamic_trusted, dynamic_threshold = evaluator.get_trusted_clients_dynamic(
            trust_scores, round_num, global_accuracy
        )
        
        # Calculate statistics
        trust_values = list(trust_scores.values())
        trusted_ratio = len(dynamic_trusted) / len(trust_scores)
        
        thresholds.append(dynamic_threshold)
        trusted_ratios.append(trusted_ratio)
        
        # Determine adaptation type
        if round_num > 1:
            threshold_change = dynamic_threshold - thresholds[-2]
            if threshold_change > 0.02:
                adaptation = "↗️ Increasing"
            elif threshold_change < -0.02:
                adaptation = "↘️ Decreasing"
            else:
                adaptation = "→ Stable"
        else:
            adaptation = "Initial"
        
        print(f"{round_num:5d} | {dynamic_threshold:9.3f} | {len(dynamic_trusted):2d}/{len(trust_scores):2d}    | "
              f"[{np.min(trust_values):.3f}, {np.max(trust_values):.3f}] | {global_accuracy:11.3f} | {adaptation}")
        
        # Update performance history
        evaluator.update_performance_history(global_accuracy, round_num)
    
    # Show statistics
    print(f"\nThreshold Statistics:")
    print(f"  Mean: {np.mean(thresholds):.3f}")
    print(f"  Std:  {np.std(thresholds):.3f}")
    print(f"  Range: [{np.min(thresholds):.3f}, {np.max(thresholds):.3f}]")
    print(f"  Trend: {thresholds[-1] - thresholds[0]:+.3f}")
    
    print(f"\nTrusted Client Ratio Statistics:")
    print(f"  Mean: {np.mean(trusted_ratios):.3f}")
    print(f"  Std:  {np.std(trusted_ratios):.3f}")
    print(f"  Target: 0.600")

def demonstrate_configuration_impact():
    """Demonstrate impact of different configuration parameters."""
    print("\n" + "="*80)
    print("CONFIGURATION PARAMETER IMPACT")
    print("="*80)
    
    base_config = {
        'trust': {
            'mode': 'hybrid',
            'dynamic_threshold': {
                'enabled': True,
                'min_trust_threshold': 0.1,
                'max_trust_threshold': 0.9,
                'min_trusted_clients': 2,
                'target_trusted_ratio': 0.6
            },
            'weights': {'cosine': 0.4, 'entropy': 0.3, 'reputation': 0.3}
        }
    }
    
    configurations = [
        ("Conservative (target 40%)", {'target_trusted_ratio': 0.4, 'min_trusted_clients': 3}),
        ("Moderate (target 60%)", {'target_trusted_ratio': 0.6, 'min_trusted_clients': 2}),
        ("Inclusive (target 80%)", {'target_trusted_ratio': 0.8, 'min_trusted_clients': 1}),
    ]
    
    # Fixed scenario for comparison
    trust_scores = simulate_trust_scores(10, 5, "normal")
    global_accuracy = 0.75
    
    print(f"Trust scores: {[f'{v:.3f}' for v in sorted(trust_scores.values(), reverse=True)]}")
    print(f"Global accuracy: {global_accuracy:.3f}")
    print()
    
    for config_name, modifications in configurations:
        # Create modified config
        config = base_config.copy()
        config['trust']['dynamic_threshold'].update(modifications)
        
        # Initialize evaluator
        evaluator = TrustEvaluator(config)
        
        # Calculate threshold
        dynamic_trusted, dynamic_threshold = evaluator.get_trusted_clients_dynamic(
            trust_scores, 5, global_accuracy
        )
        
        trusted_ratio = len(dynamic_trusted) / len(trust_scores)
        
        print(f"{config_name}:")
        print(f"  Dynamic threshold: {dynamic_threshold:.3f}")
        print(f"  Trusted clients: {len(dynamic_trusted)}/{len(trust_scores)} ({trusted_ratio:.1%})")
        print(f"  Configuration: {modifications}")
        print()

def main():
    """Main demonstration function."""
    print("TRUST_MCNet Dynamic Threshold Mechanism Demonstration")
    print("This demonstrates the new adaptive trust threshold system")
    print("that replaces static thresholds with intelligent, context-aware calculation.")
    
    try:
        # Demonstrate static vs dynamic comparison
        demonstrate_static_vs_dynamic()
        
        # Demonstrate adaptation over time
        demonstrate_adaptation_over_time()
        
        # Demonstrate configuration impact
        demonstrate_configuration_impact()
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("✅ Dynamic threshold successfully implemented")
        print("✅ Adapts to trust score distributions")
        print("✅ Considers round progression and performance history")
        print("✅ Ensures minimum trusted clients")
        print("✅ Preserves existing trust evaluation algorithms")
        print("✅ Configurable via trust.yaml settings")
        
        print("\nKey Benefits:")
        print("- Prevents 'no trusted clients' scenarios")
        print("- Adapts to changing client trust patterns")
        print("- Maintains federated learning progress")
        print("- Balances security and participation")
        print("- Provides comprehensive logging and statistics")
        
        print("\nTo use in your federated learning setup:")
        print("1. Set 'trust.dynamic_threshold.enabled: true' in config")
        print("2. Call evaluator.get_trusted_clients_dynamic() instead of static filtering")
        print("3. Monitor threshold evolution via evaluator.get_threshold_statistics()")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
