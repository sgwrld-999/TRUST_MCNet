"""
Persistent Reputation Database for TRUST_MCNet

This module provides database storage for trust scores, reputation history,
and client performance metrics with SQLite backend.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
from contextlib import contextmanager


class ReputationDatabase:
    """
    Persistent storage for client trust scores and reputation data.
    
    Provides thread-safe database operations for storing and retrieving:
    - Trust scores over time
    - Reputation history
    - Performance metrics
    - Quarantine events
    """
    
    def __init__(self, db_path: str = "trust_mcnet_reputation.db"):
        """
        Initialize the reputation database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Create database and tables
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Trust scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trust_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    trust_score REAL NOT NULL,
                    trust_mode TEXT NOT NULL,
                    cosine_score REAL,
                    entropy_score REAL,
                    reputation_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, round_number)
                )
            """)
            
            # Client history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    f1_score REAL,
                    participation_rate REAL,
                    flags INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, round_number)
                )
            """)
            
            # Quarantine events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarantine_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    event_type TEXT NOT NULL,  -- 'QUARANTINED' or 'RELEASED'
                    reason TEXT,
                    duration INTEGER,
                    trust_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Adaptive threshold history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER NOT NULL,
                    threshold_value REAL NOT NULL,
                    target_accuracy REAL,
                    current_accuracy REAL,
                    adaptation_reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(round_number)
                )
            """)
            
            conn.commit()
            
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
            
    def store_trust_score(
        self,
        client_id: str,
        round_number: int,
        trust_score: float,
        trust_mode: str = "hybrid",
        cosine_score: Optional[float] = None,
        entropy_score: Optional[float] = None,
        reputation_score: Optional[float] = None
    ) -> bool:
        """
        Store trust score for a client in a specific round.
        
        Args:
            client_id: Client identifier
            round_number: Training round number
            trust_score: Overall trust score
            trust_mode: Trust evaluation mode used
            cosine_score: Cosine similarity component (optional)
            entropy_score: Entropy component (optional)
            reputation_score: Reputation component (optional)
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO trust_scores 
                        (client_id, round_number, trust_score, trust_mode, 
                         cosine_score, entropy_score, reputation_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (client_id, round_number, trust_score, trust_mode,
                          cosine_score, entropy_score, reputation_score))
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to store trust score for {client_id}: {e}")
            return False
            
    def get_trust_history(
        self, 
        client_id: str, 
        last_n_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trust score history for a client.
        
        Args:
            client_id: Client identifier
            last_n_rounds: Number of recent rounds to retrieve (optional)
            
        Returns:
            List of trust score records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if last_n_rounds:
                    cursor.execute("""
                        SELECT * FROM trust_scores 
                        WHERE client_id = ? 
                        ORDER BY round_number DESC 
                        LIMIT ?
                    """, (client_id, last_n_rounds))
                else:
                    cursor.execute("""
                        SELECT * FROM trust_scores 
                        WHERE client_id = ? 
                        ORDER BY round_number ASC
                    """, (client_id,))
                    
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get trust history for {client_id}: {e}")
            return []
            
    def store_client_performance(
        self,
        client_id: str,
        round_number: int,
        accuracy: float,
        loss: float,
        f1_score: Optional[float] = None,
        participation_rate: float = 1.0,
        flags: int = 0
    ) -> bool:
        """
        Store client performance metrics.
        
        Args:
            client_id: Client identifier
            round_number: Training round number
            accuracy: Model accuracy
            loss: Training loss
            f1_score: F1 score (optional)
            participation_rate: Client participation rate
            flags: Number of anomaly flags
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO client_history 
                        (client_id, round_number, accuracy, loss, f1_score, 
                         participation_rate, flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (client_id, round_number, accuracy, loss, f1_score,
                          participation_rate, flags))
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to store performance for {client_id}: {e}")
            return False
            
    def record_quarantine_event(
        self,
        client_id: str,
        round_number: int,
        event_type: str,
        reason: str = "",
        duration: Optional[int] = None,
        trust_score: Optional[float] = None
    ) -> bool:
        """
        Record a quarantine event (quarantine or release).
        
        Args:
            client_id: Client identifier
            round_number: Round number when event occurred
            event_type: 'QUARANTINED' or 'RELEASED'
            reason: Reason for the event
            duration: Quarantine duration in rounds (for QUARANTINED events)
            trust_score: Trust score at time of event
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO quarantine_events 
                        (client_id, round_number, event_type, reason, duration, trust_score)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (client_id, round_number, event_type, reason, duration, trust_score))
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to record quarantine event for {client_id}: {e}")
            return False
            
    def store_threshold_adaptation(
        self,
        round_number: int,
        threshold_value: float,
        target_accuracy: float,
        current_accuracy: float,
        adaptation_reason: str = ""
    ) -> bool:
        """
        Store adaptive threshold change.
        
        Args:
            round_number: Round number when adaptation occurred
            threshold_value: New threshold value
            target_accuracy: Target accuracy for adaptation
            current_accuracy: Current system accuracy
            adaptation_reason: Reason for adaptation
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO threshold_history 
                        (round_number, threshold_value, target_accuracy, 
                         current_accuracy, adaptation_reason)
                        VALUES (?, ?, ?, ?, ?)
                    """, (round_number, threshold_value, target_accuracy,
                          current_accuracy, adaptation_reason))
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to store threshold adaptation: {e}")
            return False
            
    def get_quarantine_history(self, client_id: str) -> List[Dict[str, Any]]:
        """
        Get quarantine history for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            List of quarantine event records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM quarantine_events 
                    WHERE client_id = ? 
                    ORDER BY round_number ASC
                """, (client_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get quarantine history for {client_id}: {e}")
            return []
            
    def get_all_clients_latest_trust(self) -> Dict[str, Dict[str, Any]]:
        """
        Get latest trust scores for all clients.
        
        Returns:
            Dictionary mapping client_id to latest trust score record
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t1.* FROM trust_scores t1
                    INNER JOIN (
                        SELECT client_id, MAX(round_number) as max_round
                        FROM trust_scores 
                        GROUP BY client_id
                    ) t2 ON t1.client_id = t2.client_id AND t1.round_number = t2.max_round
                """)
                
                result = {}
                for row in cursor.fetchall():
                    result[row['client_id']] = dict(row)
                return result
        except Exception as e:
            self.logger.error(f"Failed to get all clients latest trust: {e}")
            return {}
            
    def get_trust_statistics(self) -> Dict[str, Any]:
        """
        Get overall trust statistics.
        
        Returns:
            Dictionary with trust statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT client_id) as total_clients,
                        COUNT(*) as total_evaluations,
                        AVG(trust_score) as mean_trust,
                        MIN(trust_score) as min_trust,
                        MAX(trust_score) as max_trust,
                        MAX(round_number) as latest_round
                    FROM trust_scores
                """)
                stats = dict(cursor.fetchone())
                
                # Quarantine statistics
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT client_id) as clients_quarantined,
                        COUNT(*) as total_quarantine_events
                    FROM quarantine_events 
                    WHERE event_type = 'QUARANTINED'
                """)
                quarantine_stats = dict(cursor.fetchone())
                
                stats.update(quarantine_stats)
                return stats
        except Exception as e:
            self.logger.error(f"Failed to get trust statistics: {e}")
            return {}
            
    def cleanup_old_records(self, keep_last_n_rounds: int = 1000) -> int:
        """
        Clean up old records to maintain database performance.
        
        Args:
            keep_last_n_rounds: Number of recent rounds to keep
            
        Returns:
            Number of records deleted
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Find cutoff round
                    cursor.execute("""
                        SELECT MAX(round_number) - ? as cutoff_round
                        FROM trust_scores
                    """, (keep_last_n_rounds,))
                    cutoff_round = cursor.fetchone()[0] or 0
                    
                    if cutoff_round <= 0:
                        return 0
                        
                    # Delete old trust scores
                    cursor.execute("""
                        DELETE FROM trust_scores 
                        WHERE round_number < ?
                    """, (cutoff_round,))
                    deleted_trust = cursor.rowcount
                    
                    # Delete old client history
                    cursor.execute("""
                        DELETE FROM client_history 
                        WHERE round_number < ?
                    """, (cutoff_round,))
                    deleted_history = cursor.rowcount
                    
                    conn.commit()
                    total_deleted = deleted_trust + deleted_history
                    
                    self.logger.info(f"Cleaned up {total_deleted} old records")
                    return total_deleted
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            return 0
