"""
Custom exceptions for TRUST-MCNet federated learning framework.

This module defines custom exceptions that provide meaningful error messages
and enable proper error handling throughout the application.
"""

from typing import Optional, Any, Dict


class TrustMCNetError(Exception):
    """Base exception for all TRUST-MCNet related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(TrustMCNetError):
    """Raised when there are configuration-related issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 expected_type: Optional[type] = None, actual_value: Optional[Any] = None):
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value
        
        context = {
            "config_key": config_key,
            "expected_type": expected_type.__name__ if expected_type else None,
            "actual_value": actual_value
        }
        
        super().__init__(message, "CONFIG_ERROR", context)


class DataLoadingError(TrustMCNetError):
    """Raised when data loading or preprocessing fails."""
    
    def __init__(self, message: str, dataset_path: Optional[str] = None, 
                 dataset_type: Optional[str] = None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        
        context = {
            "dataset_path": dataset_path,
            "dataset_type": dataset_type
        }
        
        super().__init__(message, "DATA_ERROR", context)


class ModelError(TrustMCNetError):
    """Raised when model-related operations fail."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, 
                 operation: Optional[str] = None):
        self.model_type = model_type
        self.operation = operation
        
        context = {
            "model_type": model_type,
            "operation": operation
        }
        
        super().__init__(message, "MODEL_ERROR", context)


class TrustEvaluationError(TrustMCNetError):
    """Raised when trust evaluation fails."""
    
    def __init__(self, message: str, client_id: Optional[str] = None, 
                 trust_method: Optional[str] = None):
        self.client_id = client_id
        self.trust_method = trust_method
        
        context = {
            "client_id": client_id,
            "trust_method": trust_method
        }
        
        super().__init__(message, "TRUST_ERROR", context)


class PartitioningError(TrustMCNetError):
    """Raised when data partitioning fails."""
    
    def __init__(self, message: str, partitioning_strategy: Optional[str] = None, 
                 num_clients: Optional[int] = None, dataset_size: Optional[int] = None):
        self.partitioning_strategy = partitioning_strategy
        self.num_clients = num_clients
        self.dataset_size = dataset_size
        
        context = {
            "partitioning_strategy": partitioning_strategy,
            "num_clients": num_clients,
            "dataset_size": dataset_size
        }
        
        super().__init__(message, "PARTITION_ERROR", context)


class ExperimentError(TrustMCNetError):
    """Raised when experiment execution fails."""
    
    def __init__(self, message: str, experiment_phase: Optional[str] = None, 
                 round_number: Optional[int] = None):
        self.experiment_phase = experiment_phase
        self.round_number = round_number
        
        context = {
            "experiment_phase": experiment_phase,
            "round_number": round_number
        }
        
        super().__init__(message, "EXPERIMENT_ERROR", context)


class ClientError(TrustMCNetError):
    """Raised when client-related operations fail."""
    
    def __init__(self, message: str, client_id: Optional[str] = None, 
                 operation: Optional[str] = None):
        self.client_id = client_id
        self.operation = operation
        
        context = {
            "client_id": client_id,
            "operation": operation
        }
        
        super().__init__(message, "CLIENT_ERROR", context)


class StrategyError(TrustMCNetError):
    """Raised when federated learning strategy operations fail."""
    
    def __init__(self, message: str, strategy_type: Optional[str] = None, 
                 round_number: Optional[int] = None):
        self.strategy_type = strategy_type
        self.round_number = round_number
        
        context = {
            "strategy_type": strategy_type,
            "round_number": round_number
        }
        
        super().__init__(message, "STRATEGY_ERROR", context)


class ValidationError(TrustMCNetError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, 
                 failed_checks: Optional[list] = None):
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        
        context = {
            "validation_type": validation_type,
            "failed_checks": failed_checks
        }
        
        super().__init__(message, "VALIDATION_ERROR", context)


class ResourceError(TrustMCNetError):
    """Raised when resource allocation or management fails."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 required_amount: Optional[Any] = None, available_amount: Optional[Any] = None):
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount
        
        context = {
            "resource_type": resource_type,
            "required_amount": required_amount,
            "available_amount": available_amount
        }
        
        super().__init__(message, "RESOURCE_ERROR", context)
