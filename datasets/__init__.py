"""
Dataset registry for TRUST_MCNet federated learning framework.
Provides a centralized way to register and retrieve dataset implementations.
"""

import os
from typing import Dict, Type, Any, Optional

# Global dataset registry
DATASETS = {}


def register(name: str):
    """
    Dataset registration decorator.
    
    Args:
        name: Dataset identifier (e.g., 'ton_iot', 'edge_iiot', 'medbiot')
    
    Usage:
        @register("ton_iot")
        class ToNIoTDataset:
            pass
    """
    def wrap(cls):
        DATASETS[name] = cls
        return cls
    return wrap


def get(name: str, **kwargs) -> Any:
    """
    Get dataset instance by name.
    
    Args:
        name: Registered dataset name
        **kwargs: Arguments to pass to dataset constructor
        
    Returns:
        Dataset instance
        
    Raises:
        ValueError: If dataset name not registered
    """
    # Ensure datasets are imported and registered
    _import_datasets()
    
    if name not in DATASETS:
        available = list(DATASETS.keys())
        raise ValueError(f"Dataset '{name}' not registered. Available: {available}")
    
    return DATASETS[name](**kwargs)


def get_data_root() -> str:
    """
    Get data root directory from environment variable or default.
    
    Returns:
        Data root path
    """
    return os.getenv("MCNET_DATA", "./data")


def list_datasets() -> list:
    """
    List all registered dataset names.
    
    Returns:
        List of registered dataset names
    """
    # Ensure datasets are imported and registered
    _import_datasets()
    return list(DATASETS.keys())


def _import_datasets():
    """Import all dataset modules to trigger registration."""
    try:
        # Import dataset modules to trigger @register decorators
        from . import ton_iot
        from . import edge_iiot
        from . import medbiot
    except ImportError as e:
        # Handle case where dataset modules might not be available
        pass
