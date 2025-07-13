"""
TRUST-MCNet: Federated Learning Framework with Trust Mechanisms

A modern, production-ready federated learning framework for IoT anomaly detection 
with advanced trust evaluation mechanisms.
"""

__version__ = "2.0.0"
__author__ = "TRUST-MCNet Team"
__description__ = "Federated Learning Framework with Trust Mechanisms for IoT"

from .core import *
from .clients import *
from .trust_module import *
from .models import *
from .utils import *

__all__ = [
    "FederatedClient",
    "TrustEvaluator", 
    "FederatedServer",
    "IoTDataProcessor",
    "SimulationManager"
]
