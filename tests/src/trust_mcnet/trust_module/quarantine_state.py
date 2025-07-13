"""
Quarantine state management for TRUST_MCNet trust-based client filtering.

This module implements persistent state tracking for client quarantine decisions,
maintaining SOLID principles and clean architecture.
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Set, Tuple, Any
import logging


@dataclass
class ClientQuarantineStatus:
    """
    Tracks quarantine status for a single client.
    
    Attributes:
        below_tau_streak: Number of consecutive rounds below trust threshold
        quarantine_rounds_left: Remaining rounds in quarantine (0 = not quarantined)
        total_quarantines: Total number of times this client has been quarantined
        last_quarantine_round: Round number when last quarantined
    """
    below_tau_streak: int = 0
    quarantine_rounds_left: int = 0
    total_quarantines: int = 0
    last_quarantine_round: int = -1


class QuarantineState:
    """
    Singleton-style container for managing client quarantine state.
    
    This class follows SOLID principles:
    - Single Responsibility: Manages only quarantine state
    - Open/Closed: Extensible for different quarantine policies
    - Liskov Substitution: Can be replaced by other state managers
    - Interface Segregation: Minimal, focused interface
    - Dependency Inversion: Uses abstract configuration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quarantine state manager.
        
        Args:
            config: Configuration dictionary with quarantine parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client_status: Dict[str, ClientQuarantineStatus] = defaultdict(ClientQuarantineStatus)
        self._quarantine_history: Dict[str, list] = defaultdict(list)
        
    def update_client_status(
        self, 
        client_id: str, 
        trust_score: float, 
        round_number: int,
        tau: float = 0.35,
        patience: int = 2,
        quarantine_rounds: int = 5
    ) -> None:
        """
        Update client's quarantine status based on current trust score.
        
        Args:
            client_id: Unique client identifier
            trust_score: Current trust score for the client
            round_number: Current training round
            tau: Trust threshold below which to start counting
            patience: Number of consecutive low-trust rounds before quarantine
            quarantine_rounds: Number of rounds to quarantine client
        """
        status = self._client_status[client_id]
        
        # Update below-threshold streak
        if trust_score < tau:
            status.below_tau_streak += 1
            self.logger.debug(f"Client {client_id} below tau ({trust_score:.3f} < {tau:.3f}), "
                             f"streak: {status.below_tau_streak}")
        else:
            if status.below_tau_streak > 0:
                self.logger.debug(f"Client {client_id} trust recovered ({trust_score:.3f} >= {tau:.3f}), "
                                 f"resetting streak from {status.below_tau_streak}")
            status.below_tau_streak = 0
        
        # Check if client should enter quarantine
        if status.below_tau_streak >= patience and status.quarantine_rounds_left == 0:
            status.quarantine_rounds_left = quarantine_rounds
            status.below_tau_streak = 0  # Reset streak after quarantine starts
            status.total_quarantines += 1
            status.last_quarantine_round = round_number
            
            # Log quarantine decision
            self.logger.warning(f"QUARANTINED: Client {client_id} for {quarantine_rounds} rounds "
                               f"(patience exceeded: {patience} consecutive rounds below {tau:.3f})")
            
            # Record in history
            self._quarantine_history[client_id].append({
                'round': round_number,
                'reason': 'sustained_low_trust',
                'duration': quarantine_rounds,
                'trust_score': trust_score
            })
        
        # Decrement quarantine time if client is quarantined
        if status.quarantine_rounds_left > 0:
            status.quarantine_rounds_left -= 1
            if status.quarantine_rounds_left == 0:
                self.logger.info(f"RELEASED: Client {client_id} quarantine period ended")
    
    def is_quarantined(self, client_id: str) -> bool:
        """
        Check if a client is currently quarantined.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if client is quarantined, False otherwise
        """
        return self._client_status[client_id].quarantine_rounds_left > 0
    
    def get_quarantined_clients(self) -> Set[str]:
        """
        Get set of all currently quarantined clients.
        
        Returns:
            Set of client IDs currently in quarantine
        """
        return {client_id for client_id, status in self._client_status.items() 
                if status.quarantine_rounds_left > 0}
    
    def get_client_status(self, client_id: str) -> ClientQuarantineStatus:
        """
        Get current quarantine status for a client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            ClientQuarantineStatus object
        """
        return self._client_status[client_id]
    
    def get_quarantine_statistics(self) -> Dict[str, Any]:
        """
        Get quarantine statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing quarantine statistics
        """
        total_clients = len(self._client_status)
        quarantined_clients = len(self.get_quarantined_clients())
        
        # Calculate statistics
        total_quarantines = sum(status.total_quarantines for status in self._client_status.values())
        clients_with_quarantines = sum(1 for status in self._client_status.values() 
                                     if status.total_quarantines > 0)
        
        # Streak statistics
        current_streaks = [status.below_tau_streak for status in self._client_status.values() 
                          if status.below_tau_streak > 0]
        
        return {
            'total_clients': total_clients,
            'currently_quarantined': quarantined_clients,
            'quarantine_rate': quarantined_clients / max(1, total_clients),
            'total_quarantine_events': total_quarantines,
            'clients_ever_quarantined': clients_with_quarantines,
            'active_below_tau_streaks': len(current_streaks),
            'max_current_streak': max(current_streaks) if current_streaks else 0
        }
    
    def reset_client_state(self, client_id: str) -> None:
        """
        Reset quarantine state for a specific client (useful for testing).
        
        Args:
            client_id: Unique client identifier
        """
        if client_id in self._client_status:
            del self._client_status[client_id]
        if client_id in self._quarantine_history:
            del self._quarantine_history[client_id]
        self.logger.debug(f"Reset quarantine state for client {client_id}")
    
    def clear_all_state(self) -> None:
        """
        Clear all quarantine state (useful for testing).
        """
        self._client_status.clear()
        self._quarantine_history.clear()
        self.logger.debug("Cleared all quarantine state")
