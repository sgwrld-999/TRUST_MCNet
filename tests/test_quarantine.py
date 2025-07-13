"""
Unit tests for quarantine and trimming logic hook in TRUST_MCNet.

Tests the complete quarantine workflow including:
- Client state tracking
- Quarantine decisions
- Trust-weighted aggregation with quarantine filtering
- Recovery from quarantine
"""

import pytest
import numpy as np
import torch
from typing import Dict, List
from unittest.mock import Mock, patch

# Import modules under test
try:
    from tests.src.trust_mcnet.trust_module.quarantine_state import QuarantineState, ClientQuarantineStatus
    from tests.src.trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
except ImportError:
    pytest.skip("Trust modules not available", allow_module_level=True)


class TestQuarantineState:
    """Test cases for QuarantineState class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quarantine_state = QuarantineState()
        self.test_config = {
            'tau': 0.35,
            'patience': 2,
            'quarantine_rounds': 3
        }
    
    def test_initial_state(self):
        """Test initial quarantine state is empty."""
        assert len(self.quarantine_state.get_quarantined_clients()) == 0
        assert not self.quarantine_state.is_quarantined("test_client")
    
    def test_below_threshold_tracking(self):
        """Test tracking of consecutive below-threshold rounds."""
        client_id = "test_client"
        
        # First round below threshold
        self.quarantine_state.update_client_status(
            client_id=client_id,
            trust_score=0.3,  # Below tau=0.35
            round_number=1,
            **self.test_config
        )
        
        status = self.quarantine_state.get_client_status(client_id)
        assert status.below_tau_streak == 1
        assert not self.quarantine_state.is_quarantined(client_id)
        
        # Second round below threshold - should trigger quarantine
        self.quarantine_state.update_client_status(
            client_id=client_id,
            trust_score=0.25,
            round_number=2,
            **self.test_config
        )
        
        assert self.quarantine_state.is_quarantined(client_id)
        status = self.quarantine_state.get_client_status(client_id)
        assert status.quarantine_rounds_left == 3
        assert status.total_quarantines == 1
        assert status.below_tau_streak == 0  # Reset after quarantine
    
    def test_streak_reset_on_recovery(self):
        """Test that below-threshold streak resets when trust recovers."""
        client_id = "test_client"
        
        # One round below threshold
        self.quarantine_state.update_client_status(
            client_id=client_id,
            trust_score=0.3,
            round_number=1,
            **self.test_config
        )
        
        # Trust recovers
        self.quarantine_state.update_client_status(
            client_id=client_id,
            trust_score=0.6,
            round_number=2,
            **self.test_config
        )
        
        status = self.quarantine_state.get_client_status(client_id)
        assert status.below_tau_streak == 0
        assert not self.quarantine_state.is_quarantined(client_id)
    
    def test_quarantine_countdown(self):
        """Test quarantine countdown and release."""
        client_id = "test_client"
        
        # Trigger quarantine
        for round_num in [1, 2]:
            self.quarantine_state.update_client_status(
                client_id=client_id,
                trust_score=0.2,
                round_number=round_num,
                **self.test_config
            )
        
        # Client should be quarantined
        assert self.quarantine_state.is_quarantined(client_id)
        status = self.quarantine_state.get_client_status(client_id)
        assert status.quarantine_rounds_left == 3
        
        # Countdown quarantine rounds
        for round_num in [3, 4, 5]:
            self.quarantine_state.update_client_status(
                client_id=client_id,
                trust_score=0.8,  # High trust but still quarantined
                round_number=round_num,
                **self.test_config
            )
            
            status = self.quarantine_state.get_client_status(client_id)
            expected_remaining = max(0, 3 - (round_num - 2))
            assert status.quarantine_rounds_left == expected_remaining
        
        # Should be released after quarantine period
        assert not self.quarantine_state.is_quarantined(client_id)
    
    def test_quarantine_statistics(self):
        """Test quarantine statistics collection."""
        # Add multiple clients with different states
        clients = ["client_1", "client_2", "client_3"]
        
        # Quarantine client_1
        for round_num in [1, 2]:
            self.quarantine_state.update_client_status(
                client_id="client_1",
                trust_score=0.2,
                round_number=round_num,
                **self.test_config
            )
        
        # Client_2 has low trust but not quarantined yet
        self.quarantine_state.update_client_status(
            client_id="client_2",
            trust_score=0.3,
            round_number=1,
            **self.test_config
        )
        
        # Client_3 is healthy
        self.quarantine_state.update_client_status(
            client_id="client_3",
            trust_score=0.8,
            round_number=1,
            **self.test_config
        )
        
        stats = self.quarantine_state.get_quarantine_statistics()
        
        assert stats['total_clients'] == 3
        assert stats['currently_quarantined'] == 1
        assert stats['quarantine_rate'] == 1/3
        assert stats['total_quarantine_events'] == 1
        assert stats['clients_ever_quarantined'] == 1
        assert stats['active_below_tau_streaks'] == 1
        assert stats['max_current_streak'] == 1


class TestTrustEvaluatorQuarantine:
    """Test cases for TrustEvaluator quarantine integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'trust': {
                'quarantine': {
                    'tau': 0.35,
                    'patience': 2,
                    'quarantine_rounds': 3,
                    'enable_quarantine': True
                },
                'aggregation': {
                    'trim_ratio': 0.2,
                    'min_clients_for_trimming': 3
                }
            }
        }
        
        self.trust_evaluator = TrustEvaluator(
            trust_mode='hybrid',
            threshold=0.5,
            config=self.config
        )
    
    def test_detect_malicious_clients_basic(self):
        """Test basic malicious client detection."""
        client_ids = ["good_client", "bad_client"]
        trust_scores = [0.8, 0.2]  # bad_client below tau
        
        quarantined, survivors = self.trust_evaluator.detect_malicious_clients(
            client_ids=client_ids,
            trust_vec=trust_scores,
            round_number=1
        )
        
        # First round: no quarantine yet
        assert quarantined == []
        assert set(survivors) == set(client_ids)
        
        # Second round: bad_client should be quarantined
        quarantined, survivors = self.trust_evaluator.detect_malicious_clients(
            client_ids=client_ids,
            trust_vec=trust_scores,
            round_number=2
        )
        
        assert "bad_client" in quarantined
        assert "good_client" in survivors
        assert "bad_client" not in survivors
    
    def test_quarantine_disabled(self):
        """Test behavior when quarantine is disabled."""
        config_no_quarantine = {
            'trust': {
                'quarantine': {
                    'enable_quarantine': False
                }
            }
        }
        
        trust_evaluator = TrustEvaluator(config=config_no_quarantine)
        
        client_ids = ["client_1", "client_2"]
        trust_scores = [0.1, 0.1]  # Both very low trust
        
        quarantined, survivors = trust_evaluator.detect_malicious_clients(
            client_ids=client_ids,
            trust_vec=trust_scores,
            round_number=1
        )
        
        # With quarantine disabled, all clients survive
        assert quarantined == []
        assert set(survivors) == set(client_ids)
    
    def test_aggregate_with_quarantine(self):
        """Test model aggregation with quarantine logic."""
        # Create dummy client updates
        client_updates = {
            "good_client": {
                "param_0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "param_1": torch.tensor([0.5, 1.5], dtype=torch.float32)
            },
            "bad_client": {
                "param_0": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),  # Outlier
                "param_1": torch.tensor([5.0, 15.0], dtype=torch.float32)  # Outlier
            },
            "neutral_client": {
                "param_0": torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
                "param_1": torch.tensor([1.0, 2.0], dtype=torch.float32)
            }
        }
        
        # Trust scores: bad_client will be quarantined
        trust_scores = {
            "good_client": 0.8,
            "bad_client": 0.2,   # Below tau, will be quarantined after patience rounds
            "neutral_client": 0.6
        }
        
        # First round aggregation - no quarantine yet
        aggregated_model, trust_stats = self.trust_evaluator.aggregate_model_updates(
            client_updates=client_updates,
            client_trust_scores=trust_scores,
            round_number=1
        )
        
        assert len(trust_stats['quarantined_clients']) == 0
        assert len(trust_stats['surviving_clients']) == 3
        
        # Second round - bad_client should be quarantined
        aggregated_model, trust_stats = self.trust_evaluator.aggregate_model_updates(
            client_updates=client_updates,
            client_trust_scores=trust_scores,
            round_number=2
        )
        
        assert "bad_client" in trust_stats['quarantined_clients']
        assert "bad_client" not in trust_stats['surviving_clients']
        assert len(trust_stats['surviving_clients']) == 2
        
        # Verify aggregated model excludes quarantined client
        assert "param_0" in aggregated_model
        assert "param_1" in aggregated_model
        
        # The aggregated parameters should be closer to good_client + neutral_client
        # without the outlier bad_client
        assert aggregated_model["param_0"].shape == torch.Size([2, 2])
        assert aggregated_model["param_1"].shape == torch.Size([2])
    
    def test_quarantine_recovery(self):
        """Test client recovery from quarantine."""
        client_ids = ["recovering_client"]
        
        # Trigger quarantine with low trust scores
        for round_num in [1, 2]:
            self.trust_evaluator.detect_malicious_clients(
                client_ids=client_ids,
                trust_vec=[0.2],
                round_number=round_num
            )
        
        # Client should be quarantined
        assert self.trust_evaluator.quarantine_state.is_quarantined("recovering_client")
        
        # Wait out quarantine period with good trust
        for round_num in [3, 4, 5]:
            quarantined, survivors = self.trust_evaluator.detect_malicious_clients(
                client_ids=client_ids,
                trust_vec=[0.9],  # High trust
                round_number=round_num
            )
        
        # Client should be released
        assert not self.trust_evaluator.quarantine_state.is_quarantined("recovering_client")
        assert "recovering_client" in survivors
        assert "recovering_client" not in quarantined


class TestQuarantineIntegration:
    """Integration tests for complete quarantine workflow."""
    
    def test_quarantine_cycle_end_to_end(self):
        """Test complete quarantine cycle from detection to recovery."""
        config = {
            'trust': {
                'quarantine': {
                    'tau': 0.35,
                    'patience': 2,
                    'quarantine_rounds': 3,
                    'enable_quarantine': True
                },
                'aggregation': {
                    'trim_ratio': 0.1
                }
            }
        }
        
        trust_evaluator = TrustEvaluator(config=config)
        
        # Create mock client updates
        good_update = {
            "param_0": torch.tensor([1.0, 2.0], dtype=torch.float32)
        }
        bad_update = {
            "param_0": torch.tensor([100.0, 200.0], dtype=torch.float32)  # Outlier
        }
        
        client_updates = {
            "good_client": good_update,
            "bad_client": bad_update
        }
        
        # Simulate training rounds
        round_results = []
        
        for round_num in range(1, 8):
            trust_scores = {
                "good_client": 0.8,
                "bad_client": 0.2 if round_num <= 4 else 0.9  # Recovery after round 4
            }
            
            try:
                aggregated_model, trust_stats = trust_evaluator.aggregate_model_updates(
                    client_updates=client_updates,
                    client_trust_scores=trust_scores,
                    round_number=round_num
                )
                
                round_results.append({
                    'round': round_num,
                    'quarantined': trust_stats['quarantined_clients'],
                    'survivors': trust_stats['surviving_clients'],
                    'num_quarantined': trust_stats['num_quarantined'],
                    'aggregated_shape': aggregated_model["param_0"].shape if aggregated_model else None
                })
                
            except Exception as e:
                round_results.append({
                    'round': round_num,
                    'error': str(e)
                })
        
        # Verify quarantine pattern
        # Rounds 1-2: Building up to quarantine
        assert round_results[0]['num_quarantined'] == 0  # Round 1
        assert round_results[1]['num_quarantined'] == 1  # Round 2: quarantine triggered
        
        # Rounds 3-5: bad_client quarantined
        for i in [2, 3, 4]:  # Rounds 3-5
            assert "bad_client" in round_results[i]['quarantined']
            assert "bad_client" not in round_results[i]['survivors']
        
        # Rounds 6+: bad_client released and recovered
        assert round_results[5]['num_quarantined'] == 0  # Round 6
        assert "bad_client" in round_results[5]['survivors']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
