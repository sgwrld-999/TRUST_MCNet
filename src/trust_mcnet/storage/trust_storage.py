"""
Trust Storage Integration Layer

Provides high-level interface for persistent trust and reputation storage
that integrates with the existing TrustEvaluator.
"""

import logging
from typing import Dict, List, Optional, Any
from .reputation_db import ReputationDatabase


class TrustStorage:
    """
    High-level interface for trust and reputation storage.
    
    Provides easy integration with existing TrustEvaluator while
    adding persistent storage capabilities.
    """
    
    def __init__(self, db_path: str = "trust_mcnet_reputation.db"):
        """
        Initialize trust storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db = ReputationDatabase(db_path)
        self.logger = logging.getLogger(__name__)
        
    def save_trust_evaluation(
        self,
        client_id: str,
        round_number: int,
        trust_data: Dict[str, Any]
    ) -> bool:
        """
        Save complete trust evaluation data.
        
        Args:
            client_id: Client identifier
            round_number: Training round number
            trust_data: Dictionary containing trust evaluation results
                       Expected keys: trust_score, trust_mode, cosine_score,
                       entropy_score, reputation_score, performance_metrics
                       
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Store trust scores
            trust_stored = self.db.store_trust_score(
                client_id=client_id,
                round_number=round_number,
                trust_score=trust_data.get('trust_score', 0.0),
                trust_mode=trust_data.get('trust_mode', 'hybrid'),
                cosine_score=trust_data.get('cosine_score'),
                entropy_score=trust_data.get('entropy_score'),
                reputation_score=trust_data.get('reputation_score')
            )
            
            # Store performance metrics if available
            performance_stored = True
            if 'performance_metrics' in trust_data:
                metrics = trust_data['performance_metrics']
                performance_stored = self.db.store_client_performance(
                    client_id=client_id,
                    round_number=round_number,
                    accuracy=metrics.get('accuracy', 0.0),
                    loss=metrics.get('loss', 1.0),
                    f1_score=metrics.get('f1_score'),
                    participation_rate=metrics.get('participation_rate', 1.0),
                    flags=metrics.get('flags', 0)
                )
                
            return trust_stored and performance_stored
            
        except Exception as e:
            self.logger.error(f"Failed to save trust evaluation for {client_id}: {e}")
            return False
            
    def load_client_trust_history(
        self,
        client_id: str,
        last_n_rounds: Optional[int] = None
    ) -> List[float]:
        """
        Load trust score history for a client.
        
        Args:
            client_id: Client identifier
            last_n_rounds: Number of recent rounds to load
            
        Returns:
            List of trust scores in chronological order
        """
        try:
            history = self.db.get_trust_history(client_id, last_n_rounds)
            return [record['trust_score'] for record in history]
        except Exception as e:
            self.logger.error(f"Failed to load trust history for {client_id}: {e}")
            return []
            
    def load_all_clients_current_trust(self) -> Dict[str, float]:
        """
        Load current trust scores for all clients.
        
        Returns:
            Dictionary mapping client_id to current trust score
        """
        try:
            latest_trust = self.db.get_all_clients_latest_trust()
            return {
                client_id: data['trust_score'] 
                for client_id, data in latest_trust.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to load all clients current trust: {e}")
            return {}
            
    def record_quarantine(
        self,
        client_id: str,
        round_number: int,
        is_quarantined: bool,
        reason: str = "",
        duration: Optional[int] = None,
        trust_score: Optional[float] = None
    ) -> bool:
        """
        Record quarantine event.
        
        Args:
            client_id: Client identifier
            round_number: Current round number
            is_quarantined: True if being quarantined, False if released
            reason: Reason for quarantine/release
            duration: Quarantine duration (for quarantine events)
            trust_score: Current trust score
            
        Returns:
            True if recorded successfully, False otherwise
        """
        event_type = "QUARANTINED" if is_quarantined else "RELEASED"
        return self.db.record_quarantine_event(
            client_id=client_id,
            round_number=round_number,
            event_type=event_type,
            reason=reason,
            duration=duration,
            trust_score=trust_score
        )
        
    def record_threshold_change(
        self,
        round_number: int,
        new_threshold: float,
        target_accuracy: float,
        current_accuracy: float,
        reason: str = ""
    ) -> bool:
        """
        Record adaptive threshold change.
        
        Args:
            round_number: Round when change occurred
            new_threshold: New threshold value
            target_accuracy: Target accuracy
            current_accuracy: Current accuracy
            reason: Reason for change
            
        Returns:
            True if recorded successfully, False otherwise
        """
        return self.db.store_threshold_adaptation(
            round_number=round_number,
            threshold_value=new_threshold,
            target_accuracy=target_accuracy,
            current_accuracy=current_accuracy,
            adaptation_reason=reason
        )
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        return self.db.get_trust_statistics()
        
    def cleanup_storage(self, keep_rounds: int = 1000) -> int:
        """
        Clean up old storage records.
        
        Args:
            keep_rounds: Number of recent rounds to keep
            
        Returns:
            Number of records deleted
        """
        return self.db.cleanup_old_records(keep_rounds)
