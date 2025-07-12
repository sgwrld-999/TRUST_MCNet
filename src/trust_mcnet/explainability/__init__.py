"""
Enhanced explainability module for TRUST_MCNet with comprehensive SHAP integration.

This module provides state-of-the-art explainability capabilities for the TRUST_MCNet
federated learning framework, including:

- Multi-model SHAP explanations (PyTorch, XGBoost, RandomForest, IsolationForest)
- Trust attribution for federated learning clients
- Performance-optimized explanations with caching
- Comprehensive visualization suite
- Integration with alerting mechanisms

Key Components:
- EnhancedSHAPExplainer: Core explainer supporting multiple model types
- AnomalyExplanationPipeline: End-to-end explanation workflow
- TrustAttributionEngine: Client trust analysis for federated learning
- SHAPVisualizationManager: Comprehensive visualization capabilities
- XGBoostExplainer, RandomForestExplainer, IsolationForestExplainer: Traditional ML model wrappers
"""

# Import availability checks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Core explainer imports
from .shap_explainer import (
    EnhancedSHAPExplainer,
    AnomalyExplanationPipeline
)

# Model wrapper imports (conditional on availability)
if XGBOOST_AVAILABLE and SKLEARN_AVAILABLE:
    from .model_wrappers import (
        XGBoostExplainer,
        RandomForestExplainer,
        IsolationForestExplainer,
        MLModelWrapper
    )
else:
    XGBoostExplainer = None
    RandomForestExplainer = None
    IsolationForestExplainer = None
    MLModelWrapper = None

# Trust attribution imports
from .trust_attribution import (
    TrustAttributionEngine,
    ClientExplanation,
    TrustMetrics,
    AlertingIntegration
)

# Visualization imports (conditional on matplotlib availability)
if MATPLOTLIB_AVAILABLE:
    from .visualization_manager import SHAPVisualizationManager
else:
    SHAPVisualizationManager = None

# Version and metadata
__version__ = "1.0.0"
__author__ = "TRUST_MCNet Development Team"

# Availability information
AVAILABILITY_INFO = {
    'shap': SHAP_AVAILABLE,
    'xgboost': XGBOOST_AVAILABLE,
    'sklearn': SKLEARN_AVAILABLE,
    'torch': TORCH_AVAILABLE,
    'matplotlib': MATPLOTLIB_AVAILABLE,
    'plotly': PLOTLY_AVAILABLE,
    'pandas': PANDAS_AVAILABLE
}

# Public API
__all__ = [
    # Core explainer classes
    'EnhancedSHAPExplainer',
    'AnomalyExplanationPipeline',
    
    # Model wrappers
    'XGBoostExplainer',
    'RandomForestExplainer', 
    'IsolationForestExplainer',
    'MLModelWrapper',
    
    # Trust attribution
    'TrustAttributionEngine',
    'ClientExplanation',
    'TrustMetrics',
    'AlertingIntegration',
    
    # Visualization
    'SHAPVisualizationManager',
    
    # Utilities
    'AVAILABILITY_INFO',
    'check_dependencies',
    'get_installation_commands'
]


def check_dependencies() -> dict:
    """
    Check which dependencies are available and which are missing.
    
    Returns:
        Dictionary with dependency status and installation commands
    """
    missing_deps = []
    available_deps = []
    
    deps_info = {
        'shap': ('pip install shap', SHAP_AVAILABLE),
        'xgboost': ('pip install xgboost', XGBOOST_AVAILABLE),
        'sklearn': ('pip install scikit-learn', SKLEARN_AVAILABLE),
        'torch': ('pip install torch', TORCH_AVAILABLE),
        'matplotlib': ('pip install matplotlib seaborn', MATPLOTLIB_AVAILABLE),
        'plotly': ('pip install plotly', PLOTLY_AVAILABLE),
        'pandas': ('pip install pandas', PANDAS_AVAILABLE)
    }
    
    for dep, (install_cmd, available) in deps_info.items():
        if available:
            available_deps.append(dep)
        else:
            missing_deps.append((dep, install_cmd))
    
    return {
        'available': available_deps,
        'missing': missing_deps,
        'all_available': len(missing_deps) == 0
    }


def get_installation_commands() -> list:
    """
    Get pip installation commands for missing dependencies.
    
    Returns:
        List of pip install commands for missing dependencies
    """
    deps = check_dependencies()
    return [cmd for _, cmd in deps['missing']]
