"""
TRUST-MCNet Server Module

This module provides various server implementations for federated learning:
- FederatedServer: Base server with trust evaluation
- FLTrustFederatedServer: Enhanced server with FLTrust aggregation support
"""

from .server import FederatedServer
from .fltrust_server import FLTrustFederatedServer, create_fltrust_server

__all__ = [
    'FederatedServer',
    'FLTrustFederatedServer', 
    'create_fltrust_server'
]