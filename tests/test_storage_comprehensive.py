"""
Comprehensive unit tests for TRUST_MCNet storage layer.

Tests the persistent reputation database and trust storage functionality.
"""

import unittest
import tempfile
import sqlite3
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import modules to test
from src.trust_mcnet.storage.reputation_db import ReputationDatabase
from src.trust_mcnet.storage.trust_storage import TrustStorage


class TestReputationDatabase(unittest.TestCase):
    """Test cases for ReputationDatabase class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize database
        self.db = ReputationDatabase(db_path=self.db_path)
        
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connection
        if hasattr(self.db, '_connection') and self.db._connection:
            self.db._connection.close()
            
        # Remove temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
            
    def test_database_initialization(self):
        """Test that database is properly initialized with correct schema."""
        # Check that tables exist
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check trust_scores table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trust_scores'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check client_history table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='client_history'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check quarantine_events table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quarantine_events'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check threshold_history table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='threshold_history'")
            self.assertIsNotNone(cursor.fetchone())
            
    def test_store_trust_score(self):
        """Test storing trust scores."""
        # Store a trust score
        self.db.store_trust_score(
            client_id="client_1",
            round_number=1,
            trust_score=0.85,
            accuracy=0.90,
            loss=0.25,
            model_quality=0.88
        )
        
        # Retrieve and verify
        history = self.db.get_trust_history("client_1", limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['client_id'], "client_1")
        self.assertEqual(history[0]['trust_score'], 0.85)
        self.assertEqual(history[0]['accuracy'], 0.90)
        
    def test_get_latest_trust(self):
        """Test retrieving latest trust scores."""
        # Store multiple trust scores
        clients = ["client_1", "client_2", "client_3"]
        for i, client_id in enumerate(clients):
            self.db.store_trust_score(
                client_id=client_id,
                round_number=1,
                trust_score=0.5 + i * 0.1,
                accuracy=0.8 + i * 0.05,
                loss=0.3 - i * 0.05
            )
            
        # Get latest trust scores
        latest = self.db.get_latest_trust_scores()
        
        self.assertEqual(len(latest), 3)
        self.assertIn("client_1", latest)
        self.assertIn("client_2", latest)
        self.assertIn("client_3", latest)
        self.assertEqual(latest["client_1"], 0.5)
        self.assertEqual(latest["client_2"], 0.6)
        
    def test_record_quarantine_event(self):
        """Test recording quarantine events."""
        # Record quarantine event
        self.db.record_quarantine_event(
            client_id="client_1",
            round_number=5,
            event_type="QUARANTINED",
            reason="Low trust score"
        )
        
        # Retrieve and verify
        events = self.db.get_quarantine_history("client_1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], "QUARANTINED")
        self.assertEqual(events[0]['reason'], "Low trust score")
        
    def test_record_threshold_change(self):
        """Test recording threshold changes."""
        # Record threshold change
        self.db.record_threshold_change(
            round_number=10,
            new_threshold=0.7,
            target_accuracy=0.85,
            current_accuracy=0.82,
            reason="Adaptation step"
        )
        
        # Retrieve and verify
        history = self.db.get_threshold_history(limit=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['new_threshold'], 0.7)
        self.assertEqual(history[0]['target_accuracy'], 0.85)
        
    def test_get_trust_statistics(self):
        """Test calculating trust statistics."""
        # Store trust scores for multiple clients
        clients_data = [
            ("client_1", 0.8, 0.85),
            ("client_2", 0.6, 0.75),
            ("client_3", 0.9, 0.95),
            ("client_4", 0.4, 0.65)
        ]
        
        for client_id, trust, accuracy in clients_data:
            self.db.store_trust_score(
                client_id=client_id,
                round_number=1,
                trust_score=trust,
                accuracy=accuracy,
                loss=1.0 - accuracy
            )
            
        # Get statistics
        stats = self.db.get_trust_statistics()
        
        self.assertEqual(stats['total_clients'], 4)
        self.assertAlmostEqual(stats['mean_trust'], 0.675, places=3)
        self.assertEqual(stats['min_trust'], 0.4)
        self.assertEqual(stats['max_trust'], 0.9)
        
    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # Store data for multiple rounds
        for round_num in range(1, 51):  # 50 rounds
            self.db.store_trust_score(
                client_id="client_1",
                round_number=round_num,
                trust_score=0.5 + round_num * 0.01,
                accuracy=0.8,
                loss=0.2
            )
            
        # Cleanup keeping only last 30 rounds
        deleted = self.db.cleanup_old_data(keep_rounds=30)
        self.assertGreater(deleted, 0)
        
        # Verify only recent data remains
        history = self.db.get_trust_history("client_1", limit=100)
        self.assertLessEqual(len(history), 30)
        
    def test_thread_safety(self):
        """Test thread safety of database operations."""
        import threading
        import time
        
        def store_data(client_id, start_round):
            for i in range(10):
                self.db.store_trust_score(
                    client_id=client_id,
                    round_number=start_round + i,
                    trust_score=0.5 + i * 0.01,
                    accuracy=0.8,
                    loss=0.2
                )
                time.sleep(0.001)  # Small delay
                
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=store_data, args=(f"client_{i}", i * 10))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify all data was stored correctly
        all_trust = self.db.get_latest_trust_scores()
        self.assertEqual(len(all_trust), 5)
        
    def test_reset_database(self):
        """Test database reset functionality."""
        # Store some data
        self.db.store_trust_score("client_1", 1, 0.8, 0.9, 0.1)
        self.db.record_quarantine_event("client_1", 1, "QUARANTINED", "Test")
        
        # Verify data exists
        trust_data = self.db.get_latest_trust_scores()
        quarantine_data = self.db.get_quarantine_history("client_1")
        self.assertGreater(len(trust_data), 0)
        self.assertGreater(len(quarantine_data), 0)
        
        # Reset database
        self.db.reset_database()
        
        # Verify data is cleared
        trust_data = self.db.get_latest_trust_scores()
        quarantine_data = self.db.get_quarantine_history("client_1")
        self.assertEqual(len(trust_data), 0)
        self.assertEqual(len(quarantine_data), 0)


class TestTrustStorage(unittest.TestCase):
    """Test cases for TrustStorage class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize storage
        self.storage = TrustStorage(db_path=self.db_path)
        
        # Mock trust evaluator
        self.mock_evaluator = Mock()
        self.mock_evaluator.threshold = 0.5
        
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connection
        if hasattr(self.storage.db, '_connection') and self.storage.db._connection:
            self.storage.db._connection.close()
            
        # Remove temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
            
    def test_save_trust_evaluation(self):
        """Test saving trust evaluation results."""
        # Mock evaluation result
        evaluation_result = {
            'client_1': {'trust_score': 0.85, 'accuracy': 0.90, 'loss': 0.25},
            'client_2': {'trust_score': 0.60, 'accuracy': 0.75, 'loss': 0.40}
        }
        
        # Save evaluation
        self.storage.save_trust_evaluation(evaluation_result, round_number=1)
        
        # Verify data was saved
        trust_scores = self.storage.load_all_clients_current_trust()
        self.assertEqual(trust_scores['client_1'], 0.85)
        self.assertEqual(trust_scores['client_2'], 0.60)
        
    def test_load_client_trust_history(self):
        """Test loading client trust history."""
        # Store multiple rounds of data
        for round_num in range(1, 6):
            self.storage.save_trust_evaluation({
                'client_1': {
                    'trust_score': 0.5 + round_num * 0.05,
                    'accuracy': 0.8,
                    'loss': 0.2
                }
            }, round_number=round_num)
            
        # Load history
        history = self.storage.load_client_trust_history('client_1', rounds=5)
        
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0], 0.55)  # First round
        self.assertEqual(history[-1], 0.75)  # Last round
        
    def test_record_quarantine(self):
        """Test recording quarantine events."""
        # Record quarantine
        self.storage.record_quarantine(
            client_id="client_1",
            round_number=5,
            is_quarantined=True,
            reason="Trust below threshold"
        )
        
        # Verify record
        events = self.storage.db.get_quarantine_history("client_1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], "QUARANTINED")
        
    def test_record_threshold_change(self):
        """Test recording threshold changes."""
        # Record threshold change
        self.storage.record_threshold_change(
            round_number=10,
            new_threshold=0.7,
            target_accuracy=0.85,
            current_accuracy=0.82,
            reason="Adaptive adjustment"
        )
        
        # Verify record
        history = self.storage.db.get_threshold_history(limit=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['new_threshold'], 0.7)
        
    def test_get_storage_stats(self):
        """Test getting storage statistics."""
        # Add some test data
        test_data = {
            'client_1': {'trust_score': 0.8, 'accuracy': 0.85, 'loss': 0.15},
            'client_2': {'trust_score': 0.6, 'accuracy': 0.75, 'loss': 0.25},
            'client_3': {'trust_score': 0.9, 'accuracy': 0.95, 'loss': 0.05}
        }
        self.storage.save_trust_evaluation(test_data, round_number=1)
        
        # Get stats
        stats = self.storage.get_storage_stats()
        
        self.assertEqual(stats['total_clients'], 3)
        self.assertAlmostEqual(stats['mean_trust'], 0.767, places=2)
        self.assertEqual(stats['min_trust'], 0.6)
        self.assertEqual(stats['max_trust'], 0.9)
        
    def test_integration_with_trust_evaluator(self):
        """Test integration with mock trust evaluator."""
        # Create evaluation data
        evaluation_result = {
            'client_1': {'trust_score': 0.85, 'accuracy': 0.90, 'loss': 0.25},
            'client_2': {'trust_score': 0.30, 'accuracy': 0.65, 'loss': 0.55}  # Low trust
        }
        
        # Save evaluation
        self.storage.save_trust_evaluation(evaluation_result, round_number=1)
        
        # Simulate quarantine decision (client_2 has trust < threshold)
        if evaluation_result['client_2']['trust_score'] < self.mock_evaluator.threshold:
            self.storage.record_quarantine(
                client_id='client_2',
                round_number=1,
                is_quarantined=True,
                reason=f"Trust {evaluation_result['client_2']['trust_score']} below threshold {self.mock_evaluator.threshold}"
            )
            
        # Verify quarantine was recorded
        quarantine_events = self.storage.db.get_quarantine_history('client_2')
        self.assertEqual(len(quarantine_events), 1)
        self.assertEqual(quarantine_events[0]['event_type'], 'QUARANTINED')
        
        # Verify no quarantine for client_1 (above threshold)
        quarantine_events_1 = self.storage.db.get_quarantine_history('client_1')
        self.assertEqual(len(quarantine_events_1), 0)


class TestStorageEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for storage layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
            
    def test_invalid_db_path(self):
        """Test handling of invalid database path."""
        invalid_path = "/invalid/path/database.db"
        
        # Should handle gracefully or raise appropriate exception
        with self.assertRaises((OSError, sqlite3.OperationalError)):
            db = ReputationDatabase(db_path=invalid_path)
            db.store_trust_score("client_1", 1, 0.5, 0.8, 0.2)
            
    def test_empty_data_handling(self):
        """Test handling of empty data requests."""
        storage = TrustStorage(db_path=self.db_path)
        
        # Test empty client history
        history = storage.load_client_trust_history("nonexistent_client", rounds=10)
        self.assertEqual(history, [])
        
        # Test empty trust scores
        trust_scores = storage.load_all_clients_current_trust()
        self.assertEqual(trust_scores, {})
        
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        storage = TrustStorage(db_path=self.db_path)
        
        # Store large amount of data
        large_evaluation = {}
        for i in range(1000):  # 1000 clients
            large_evaluation[f"client_{i}"] = {
                'trust_score': 0.5 + (i % 50) * 0.01,
                'accuracy': 0.8 + (i % 20) * 0.01,
                'loss': 0.2 - (i % 20) * 0.005
            }
            
        # Should handle large data efficiently
        storage.save_trust_evaluation(large_evaluation, round_number=1)
        
        # Verify data integrity
        all_trust = storage.load_all_clients_current_trust()
        self.assertEqual(len(all_trust), 1000)
        
    def test_concurrent_access(self):
        """Test concurrent database access."""
        import threading
        import random
        
        storage = TrustStorage(db_path=self.db_path)
        errors = []
        
        def concurrent_operation(thread_id):
            try:
                for i in range(10):
                    evaluation = {
                        f"client_{thread_id}_{i}": {
                            'trust_score': random.uniform(0.1, 0.9),
                            'accuracy': random.uniform(0.5, 0.95),
                            'loss': random.uniform(0.05, 0.5)
                        }
                    }
                    storage.save_trust_evaluation(evaluation, round_number=i + 1)
            except Exception as e:
                errors.append(e)
                
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Check for errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
