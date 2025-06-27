"""
TRUST MCNet - Federated Learning Framework with Trust Mechanisms

A distributed machine learning system for anomaly detection with multi-client coordination.
"""

__version__ = "1.0.0"
__author__ = "TRUST MCNet Team"
__email__ = "contact@trustmcnet.com"

from .clients.client import Client
from .models.model import MLP, LSTM
from .data.data_loader import ConfigManager

__all__ = [
    "Client",
    "MLP", 
    "LSTM",
    "ConfigManager"
]
