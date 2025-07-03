"""
Ray context manager for proper resource cleanup.

This module provides context managers and decorators for Ray initialization
and cleanup, ensuring proper resource management even on exceptions.
"""

import logging
import time
import gc
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from contextlib import contextmanager
from functools import wraps
import ray
import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RayResourceManager:
    """Manager for Ray resources with automatic cleanup."""
    
    def __init__(self, ray_config: Dict[str, Any]):
        """
        Initialize Ray resource manager.
        
        Args:
            ray_config: Ray configuration dictionary
        """
        self.ray_config = ray_config
        self.is_initialized = False
        
    def __enter__(self):
        """Enter context manager - initialize Ray."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup Ray."""
        self.cleanup()
        
        # Log any exceptions that occurred
        if exc_type is not None:
            logger.error(f"Exception occurred in Ray context: {exc_type.__name__}: {exc_val}")
    
    def initialize(self) -> None:
        """Initialize Ray with configuration."""
        if self.is_initialized or ray.is_initialized():
            logger.info("Ray already initialized, skipping")
            return
        
        try:
            logger.info(f"Initializing Ray with config: {self.ray_config}")
            
            ray.init(
                num_cpus=self.ray_config.get('num_cpus', 4),
                num_gpus=self.ray_config.get('num_gpus', 0),
                object_store_memory=self.ray_config.get('object_store_memory', 1000000000),
                dashboard_host=self.ray_config.get('dashboard_host', "127.0.0.1"),
                dashboard_port=self.ray_config.get('dashboard_port', 8265),
                ignore_reinit_error=self.ray_config.get('ignore_reinit_error', True)
            )
            
            self.is_initialized = True
            logger.info("Ray initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup Ray resources."""
        if not self.is_initialized:
            return
        
        try:
            logger.info("Cleaning up Ray resources")
            
            # Force garbage collection before shutdown
            self._cleanup_memory()
            
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed")
            
            self.is_initialized = False
            
        except Exception as e:
            logger.warning(f"Error during Ray cleanup: {e}")
        
        finally:
            # Final memory cleanup
            self._cleanup_memory()
    
    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            
            # Force garbage collection
            gc.collect()
            logger.debug("Garbage collection completed")
            
        except Exception as e:
            logger.warning(f"Error during memory cleanup: {e}")


@contextmanager
def ray_context(ray_config: Dict[str, Any]):
    """
    Context manager for Ray initialization and cleanup.
    
    Args:
        ray_config: Ray configuration dictionary
        
    Yields:
        RayResourceManager instance
        
    Example:
        with ray_context(config['env']['ray']) as ray_manager:
            # Ray operations here
            pass
        # Ray automatically cleaned up
    """
    ray_manager = RayResourceManager(ray_config)
    try:
        yield ray_manager.__enter__()
    finally:
        ray_manager.__exit__(None, None, None)


def with_ray_cleanup(ray_config_key: str = 'ray_config'):
    """
    Decorator to ensure Ray cleanup after function execution.
    
    Args:
        ray_config_key: Key to extract Ray config from function kwargs
        
    Returns:
        Decorated function with automatic Ray cleanup
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract Ray config from kwargs
            ray_config = kwargs.get(ray_config_key)
            if ray_config is None:
                # Try to extract from first argument if it's a config dict
                if args and isinstance(args[0], dict) and 'env' in args[0]:
                    ray_config = args[0]['env'].get('ray', {})
            
            if ray_config is None:
                logger.warning("No Ray config found, proceeding without Ray management")
                return func(*args, **kwargs)
            
            # Execute function with Ray management
            with ray_context(ray_config):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


@ray.remote
class ResourceMonitor:
    """Ray actor for monitoring resource usage."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.start_time = time.time()
        self.memory_snapshots = []
        
    def log_memory_usage(self) -> Dict[str, Any]:
        """Log current memory usage."""
        import psutil
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # CUDA memory (if available)
        cuda_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cuda_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i),
                    'cached': torch.cuda.memory_reserved(i)
                }
        
        snapshot = {
            'timestamp': time.time() - self.start_time,
            'system_memory_mb': system_memory.used / 1024 / 1024,
            'system_memory_percent': system_memory.percent,
            'cuda_memory': cuda_memory
        }
        
        self.memory_snapshots.append(snapshot)
        logger.debug(f"Memory usage: {snapshot['system_memory_mb']:.1f}MB "
                    f"({snapshot['system_memory_percent']:.1f}%)")
        
        return snapshot
    
    def get_memory_history(self) -> list:
        """Get memory usage history."""
        return self.memory_snapshots
    
    def cleanup_resources(self) -> None:
        """Cleanup monitored resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")


def cleanup_training_resources():
    """Cleanup training-specific resources."""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after training")
        
        # Force garbage collection
        gc.collect()
        logger.debug("Garbage collection after training")
        
    except Exception as e:
        logger.warning(f"Error during training resource cleanup: {e}")


def cleanup_evaluation_resources():
    """Cleanup evaluation-specific resources."""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after evaluation")
        
        # Force garbage collection
        gc.collect()
        logger.debug("Garbage collection after evaluation")
        
    except Exception as e:
        logger.warning(f"Error during evaluation resource cleanup: {e}")


class MemoryTracker:
    """Simple memory usage tracker."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.snapshots = []
    
    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot."""
        snapshot = {
            'label': label,
            'timestamp': time.time()
        }
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            snapshot.update({
                'system_memory_mb': memory.used / 1024 / 1024,
                'system_memory_percent': memory.percent
            })
        except ImportError:
            logger.warning("psutil not available for memory tracking")
        
        # CUDA memory
        if torch.cuda.is_available():
            snapshot['cuda_memory'] = {}
            for i in range(torch.cuda.device_count()):
                snapshot['cuda_memory'][f'gpu_{i}'] = {
                    'allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                    'cached_mb': torch.cuda.memory_reserved(i) / 1024 / 1024
                }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_memory_diff(self, start_label: str, end_label: str) -> Dict[str, Any]:
        """Get memory difference between two snapshots."""
        start_snapshot = None
        end_snapshot = None
        
        for snapshot in self.snapshots:
            if snapshot['label'] == start_label:
                start_snapshot = snapshot
            elif snapshot['label'] == end_label:
                end_snapshot = snapshot
        
        if start_snapshot is None or end_snapshot is None:
            raise ValueError(f"Could not find snapshots for labels: {start_label}, {end_label}")
        
        diff = {
            'duration': end_snapshot['timestamp'] - start_snapshot['timestamp']
        }
        
        if 'system_memory_mb' in start_snapshot and 'system_memory_mb' in end_snapshot:
            diff['system_memory_diff_mb'] = (
                end_snapshot['system_memory_mb'] - start_snapshot['system_memory_mb']
            )
        
        return diff
