"""
TRUST_MCNet API Module

REST API interface for trust management and quarantine control.

This module provides a REST API server for exposing TRUST_MCNet functionality:
- Adaptive threshold control and monitoring
- Quarantine management and status tracking  
- Trust metrics and analytics
- Configuration management
- Data export capabilities

Note: Requires FastAPI and uvicorn packages for full functionality.
Install with: pip install fastapi uvicorn
"""

try:
    from .server import TrustMCNetAPIServer
    from .endpoints import setup_api_endpoints
    API_AVAILABLE = True
except ImportError as e:
    API_AVAILABLE = False
    import logging
    logging.warning(f"API components not available: {e}")
    logging.warning("Install FastAPI and uvicorn to enable API server: pip install fastapi uvicorn")
    
    # Create mock classes for development
    class TrustMCNetAPIServer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    def setup_api_endpoints(*args, **kwargs):
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

__all__ = [
    'TrustMCNetAPIServer',
    'setup_api_endpoints',
    'API_AVAILABLE'
]
