"""
Flwr-based client implementation for TRUST-MCNet federated learning.

This module implements IoT-optimized Flwr clients with resource monitoring
and adaptive training capabilities.
"""

import logging
import time
import traceback
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import flwr as fl
from flwr.common import (
    NDArrays, 
    Parameters, 
    FitIns, 
    FitRes, 
    EvaluateIns, 
    EvaluateRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
import psutil
import threading
from collections import deque


class IoTResourceMonitor:
    """Monitor and manage IoT device resources during training."""
    
    def __init__(self, max_memory_mb: int = 512, max_cpu_percent: float = 70.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.memory_usage_history = deque(maxlen=10)
        self.cpu_usage_history = deque(maxlen=10)
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        while self.monitoring:
            try:
                # Memory usage in MB
                memory_info = psutil.virtual_memory()
                memory_used_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
                self.memory_usage_history.append(memory_used_mb)
                
                # CPU usage percentage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_history.append(cpu_percent)
                
                time.sleep(2)  # Monitor every 2 seconds
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                time.sleep(5)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            memory_info = psutil.virtual_memory()
            memory_used_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            cpu_percent = psutil.cpu_percent()
            
            return {
                "memory_mb": memory_used_mb,
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent,
                "memory_available_mb": memory_info.available / (1024 * 1024)
            }
        except Exception:
            return {"memory_mb": 0, "memory_percent": 0, "cpu_percent": 0, "memory_available_mb": 0}
    
    def is_resource_constrained(self) -> bool:
        """Check if device is resource constrained."""
        usage = self.get_current_usage()
        return (usage["memory_mb"] > self.max_memory_mb or 
                usage["cpu_percent"] > self.max_cpu_percent)
    
    def get_adaptive_batch_size(self, base_batch_size: int, min_size: int = 8, max_size: int = 64) -> int:
        """Calculate adaptive batch size based on resource usage."""
        usage = self.get_current_usage()
        
        memory_ratio = min(1.0, usage["memory_mb"] / self.max_memory_mb)
        cpu_ratio = min(1.0, usage["cpu_percent"] / self.max_cpu_percent)
        
        # Reduce batch size if high resource usage
        resource_factor = max(0.3, 1.0 - max(memory_ratio, cpu_ratio) * 0.7)
        adaptive_size = int(base_batch_size * resource_factor)
        
        return max(min_size, min(max_size, adaptive_size))


class TrustMCNetFlwrClient(fl.client.NumPyClient):
    """
    Flwr client implementation for TRUST-MCNet with IoT optimizations.
    
    Includes resource monitoring, adaptive training, and trust-aware reporting.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        config: Dict[str, Any]
    ):
        """
        Initialize TrustMCNet Flwr client.
        
        Args:
            client_id: Unique identifier for this client
            model: Neural network model
            train_dataset: Training dataset
            test_dataset: Test dataset
            config: Configuration dictionary
        """
        super().__init__()
        
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # IoT resource monitoring
        iot_config = config.get('federated', {}).get('iot_config', {})
        self.resource_monitor = IoTResourceMonitor(
            max_memory_mb=iot_config.get('max_memory_mb', 512),
            max_cpu_percent=iot_config.get('max_cpu_percent', 70)
        )
        
        # Training configuration
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking
        self.training_history = deque(maxlen=10)
        self.performance_metrics = {}
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{client_id}")
        
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """Return model parameters as NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict[str, Any]]:
        """Train the model on local data."""
        start_time = time.time()
        
        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Set model parameters
            self.set_parameters(parameters)
            
            # Configure training with IoT optimizations
            training_config = self._configure_training(config)
            
            # Perform local training
            train_loss, train_accuracy, num_examples = self._train_model(training_config)
            
            # Get updated parameters
            updated_parameters = self.get_parameters(config)
            
            # Calculate training metrics
            training_time = time.time() - start_time
            resource_usage = self._get_resource_metrics()
            
            # Performance metrics for trust evaluation
            metrics = {
                "accuracy": train_accuracy,
                "loss": train_loss,
                "training_time": training_time,
                "resource_usage": resource_usage["normalized_usage"],
                "memory_mb": resource_usage["memory_mb"],
                "cpu_percent": resource_usage["cpu_percent"],
                "num_examples": num_examples,
                "client_id": self.client_id,
                "adaptive_batch_size": training_config["effective_batch_size"]
            }
            
            # Update performance history
            self.training_history.append({
                "round": config.get("server_round", 0),
                "accuracy": train_accuracy,
                "loss": train_loss,
                "training_time": training_time
            })
            
            self.logger.info(f"Client {self.client_id} training completed. "
                           f"Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}, "
                           f"Time: {training_time:.2f}s")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"Training failed for client {self.client_id}: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return current parameters with failure metrics
            current_parameters = self.get_parameters(config)
            failure_metrics = {
                "accuracy": 0.0,
                "loss": float('inf'),
                "training_time": time.time() - start_time,
                "resource_usage": 1.0,  # High usage indicates problem
                "error": str(e),
                "client_id": self.client_id
            }
            
            return current_parameters, 0, failure_metrics
            
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on local test data."""
        try:
            # Set model parameters
            self.set_parameters(parameters)
            
            # Perform evaluation
            test_loss, test_accuracy, num_examples = self._evaluate_model(config)
            
            metrics = {
                "accuracy": test_accuracy,
                "client_id": self.client_id,
                "num_examples": num_examples
            }
            
            self.logger.info(f"Client {self.client_id} evaluation completed. "
                           f"Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
            
            return test_loss, num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for client {self.client_id}: {e}")
            return float('inf'), 0, {"accuracy": 0.0, "error": str(e), "client_id": self.client_id}
    
    def _configure_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure training parameters with IoT optimizations."""
        base_config = {
            "local_epochs": config.get("local_epochs", 5),
            "learning_rate": config.get("learning_rate", 0.001),
            "batch_size": config.get("batch_size", 32),
            "adaptive_batch_size": config.get("adaptive_batch_size", True),
            "min_batch_size": config.get("min_batch_size", 8),
            "max_batch_size": config.get("max_batch_size", 64)
        }
        
        # Adaptive batch size based on resource constraints
        if base_config["adaptive_batch_size"]:
            effective_batch_size = self.resource_monitor.get_adaptive_batch_size(
                base_config["batch_size"],
                base_config["min_batch_size"],
                base_config["max_batch_size"]
            )
        else:
            effective_batch_size = base_config["batch_size"]
        
        # Adjust epochs if resource constrained
        if self.resource_monitor.is_resource_constrained():
            base_config["local_epochs"] = max(1, base_config["local_epochs"] - 1)
        
        base_config["effective_batch_size"] = effective_batch_size
        
        # Initialize optimizer with current learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=base_config["learning_rate"]
        )
        
        return base_config
    
    def _train_model(self, training_config: Dict[str, Any]) -> Tuple[float, float, int]:
        """Perform local model training."""
        self.model.train()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config["effective_batch_size"],
            shuffle=True,
            num_workers=0  # Avoid multiprocessing on IoT devices
        )
        
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        
        for epoch in range(training_config["local_epochs"]):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Check resource constraints during training
                if batch_idx % 10 == 0 and self.resource_monitor.is_resource_constrained():
                    self.logger.warning(f"Resource constraints detected, may slow down training")
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                
                # Memory cleanup for IoT devices
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_examples += epoch_total
            
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            self.logger.debug(f"Epoch {epoch + 1}/{training_config['local_epochs']}: "
                            f"Loss: {epoch_loss / len(train_loader):.4f}, "
                            f"Accuracy: {epoch_accuracy:.4f}")
        
        avg_loss = total_loss / (training_config["local_epochs"] * len(train_loader))
        avg_accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        
        return avg_loss, avg_accuracy, total_examples
    
    def _evaluate_model(self, config: Dict[str, Any]) -> Tuple[float, float, int]:
        """Evaluate model on test data."""
        self.model.eval()
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=0
        )
        
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_examples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        
        return avg_loss, accuracy, total_examples
    
    def _get_resource_metrics(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        usage = self.resource_monitor.get_current_usage()
        
        # Normalize resource usage (0-1 scale)
        memory_ratio = min(1.0, usage["memory_mb"] / self.resource_monitor.max_memory_mb)
        cpu_ratio = min(1.0, usage["cpu_percent"] / self.resource_monitor.max_cpu_percent)
        
        normalized_usage = max(memory_ratio, cpu_ratio)
        
        return {
            "memory_mb": usage["memory_mb"],
            "cpu_percent": usage["cpu_percent"],
            "memory_percent": usage["memory_percent"],
            "normalized_usage": normalized_usage
        }


def create_flwr_client(
    client_id: str,
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any]
) -> TrustMCNetFlwrClient:
    """
    Create TrustMCNet Flwr client instance.
    
    Args:
        client_id: Unique identifier for the client
        model: Neural network model
        train_dataset: Training dataset
        test_dataset: Test dataset
        config: Configuration dictionary
        
    Returns:
        TrustMCNetFlwrClient instance
    """
    return TrustMCNetFlwrClient(
        client_id=client_id,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config
    )
