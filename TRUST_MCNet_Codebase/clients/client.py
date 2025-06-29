import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import logging
import psutil
from typing import Dict, Any, Tuple, Optional

class Client:
    """
    Enhanced client implementation with IoT device considerations.
    
    Supports both traditional federated learning and Flwr integration
    with resource monitoring and adaptive training.
    """
    
    def __init__(self, client_id, model, train_dataset, test_dataset, config):
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        # Device configuration with IoT considerations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer configuration
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['client']['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()  # Assuming classification for anomaly detection
        
        # IoT resource monitoring
        self.resource_usage_history = []
        self.performance_history = []
        
        # Adaptive training parameters
        self.iot_config = config.get('federated', {}).get('iot_config', {})
        self.max_memory_mb = self.iot_config.get('max_memory_mb', 512)
        self.max_cpu_percent = self.iot_config.get('max_cpu_percent', 70)
        self.adaptive_batch_size = self.iot_config.get('adaptive_batch_size', True)
        self.min_batch_size = self.iot_config.get('min_batch_size', 8)
        self.max_batch_size = self.iot_config.get('max_batch_size', 64)
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{client_id}")

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage for IoT monitoring."""
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
        except Exception as e:
            self.logger.warning(f"Could not get resource usage: {e}")
            return {"memory_mb": 0, "memory_percent": 0, "cpu_percent": 0, "memory_available_mb": 0}

    def is_resource_constrained(self) -> bool:
        """Check if device is resource constrained."""
        usage = self.get_resource_usage()
        return (usage["memory_mb"] > self.max_memory_mb or 
                usage["cpu_percent"] > self.max_cpu_percent)

    def get_adaptive_batch_size(self, base_batch_size: int) -> int:
        """Calculate adaptive batch size based on resource usage."""
        if not self.adaptive_batch_size:
            return base_batch_size
            
        usage = self.get_resource_usage()
        
        memory_ratio = min(1.0, usage["memory_mb"] / self.max_memory_mb)
        cpu_ratio = min(1.0, usage["cpu_percent"] / self.max_cpu_percent)
        
        # Reduce batch size if high resource usage
        resource_factor = max(0.3, 1.0 - max(memory_ratio, cpu_ratio) * 0.7)
        adaptive_size = int(base_batch_size * resource_factor)
        
        return max(self.min_batch_size, min(self.max_batch_size, adaptive_size))

    def train(self, local_epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Train the model with IoT optimizations.
        
        Args:
            local_epochs: Number of local training epochs (overrides config)
            batch_size: Batch size (overrides config)
            
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        
        # Use provided parameters or config defaults
        epochs = local_epochs or self.config['client']['local_epochs']
        base_batch_size = batch_size or self.config['client']['batch_size']
        
        # Adapt batch size for IoT constraints
        effective_batch_size = self.get_adaptive_batch_size(base_batch_size)
        
        # Adjust epochs if resource constrained
        if self.is_resource_constrained():
            epochs = max(1, epochs - 1)
            self.logger.warning(f"Resource constraints detected, reducing epochs to {epochs}")
        
        self.model.train()
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing on IoT devices
        )
        
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Monitor resources during training
                if batch_idx % 10 == 0:
                    usage = self.get_resource_usage()
                    if usage["cpu_percent"] > self.max_cpu_percent:
                        time.sleep(0.1)  # Brief pause if CPU usage is high
                
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
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_examples += epoch_total
            
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            self.logger.debug(f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}: "
                            f"Loss: {epoch_loss / len(train_loader):.4f}, "
                            f"Accuracy: {epoch_accuracy:.4f}")

        training_time = time.time() - start_time
        avg_loss = total_loss / (epochs * len(train_loader)) if len(train_loader) > 0 else 0
        avg_accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        
        # Record performance
        performance_metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "training_time": training_time,
            "effective_batch_size": effective_batch_size,
            "epochs_completed": epochs,
            "total_examples": total_examples
        }
        
        # Add resource usage
        final_usage = self.get_resource_usage()
        performance_metrics.update({
            "final_memory_mb": final_usage["memory_mb"],
            "final_cpu_percent": final_usage["cpu_percent"],
            "resource_constrained": self.is_resource_constrained()
        })
        
        self.performance_history.append(performance_metrics)
        
        self.logger.info(f"Client {self.client_id} training completed. "
                        f"Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, "
                        f"Time: {training_time:.2f}s, Batch size: {effective_batch_size}")
        
        return performance_metrics

    def test(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Test the model with IoT optimizations.
        
        Args:
            batch_size: Batch size for testing (overrides config)
            
        Returns:
            Dictionary containing test metrics
        """
        effective_batch_size = batch_size or self.config['client']['batch_size']
        
        # Adapt batch size for IoT constraints
        if self.adaptive_batch_size:
            effective_batch_size = self.get_adaptive_batch_size(effective_batch_size)
        
        self.model.eval()
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        correct = 0
        total = 0
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        
        test_metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "total_examples": total,
            "correct_predictions": correct,
            "effective_batch_size": effective_batch_size
        }
        
        self.logger.info(f"Client {self.client_id} test completed. "
                        f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return test_metrics

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def simulate_attack(self, attack_type, attack_params):
        """
        Simulates different types of attacks on the client's data or model updates.
        Args:
            attack_type (str): Type of attack ('label_flipping', 'gaussian_noise', etc.)
            attack_params (dict): Dictionary of parameters specific to the attack.
        """
        if attack_type == 'label_flipping':
            self._apply_label_flipping(attack_params)
        elif attack_type == 'gaussian_noise':
             self._apply_gaussian_noise(attack_params)
        # Add more attack types as needed
        else:
            print(f"Warning: Unknown attack type '{attack_type}' for client {self.client_id}.")

    def _apply_label_flipping(self, params):
        """
        Applies label flipping to a percentage of the training data.
        Flips anomaly labels to normal and vice versa.
        Assumes binary classification (0: normal, 1: anomaly).
        """
        flip_ratio = params.get('flip_ratio', 0.1) # Percentage of labels to flip

        if not hasattr(self.train_dataset, '__len__') or not hasattr(self.train_dataset, '__getitem__'):
            print(f"Warning: Client {self.client_id}'s train_dataset does not support label flipping.")
            return

        num_samples_to_flip = int(len(self.train_dataset) * flip_ratio)
        indices_to_flip = np.random.choice(len(self.train_dataset), num_samples_to_flip, replace=False)

        # Assuming the dataset returns (data, label) tuples
        if hasattr(self.train_dataset, 'targets'):
            for i in indices_to_flip:
                # Assuming targets is a list or numpy array of labels
                self.train_dataset.targets[i] = 1 - self.train_dataset.targets[i]
        else:
            print(f"Warning: Client {self.client_id}'s train_dataset structure is not compatible with label flipping.")


    def _apply_gaussian_noise(self, params):
        """
        Adds Gaussian noise to the model's gradients or weights after training.
        """
        noise_std = params.get('noise_std', 0.01) # Standard deviation of Gaussian noise
        noise_target = params.get('target', 'gradients') # 'gradients' or 'weights'

        if noise_target == 'gradients':
            # Add noise to gradients after calculating them
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_std
                    param.grad.data.add_(noise)
        elif noise_target == 'weights':
             # Add noise to weights after training (before sending to server)
             with torch.no_grad():
                 for param in self.model.parameters():
                     noise = torch.randn_like(param.data) * noise_std
                     param.data.add_(noise)
        else:
            print(f"Warning: Invalid noise target '{noise_target}' for client {self.client_id}.")


if __name__ == '__main__':
    # Example Usage (requires dummy dataset and model)
    print("Client class defined. Example usage requires defining dummy dataset and model.")

    # class DummyDataset(Dataset):
    #     def __init__(self, num_samples=100, input_dim=10, output_dim=2):
    #         self.data = torch.randn(num_samples, input_dim)
    #         self.targets = torch.randint(0, output_dim, (num_samples,))
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         return self.data[idx], self.targets[idx]
    #
    # class DummyModel(nn.Module):
    #     def __init__(self, input_dim=10, output_dim=2):
    #         super(DummyModel, self).__init__()
    #         self.fc = nn.Linear(input_dim, output_dim)
    #
    #     def forward(self, x):
    #         return self.fc(x)
    #
    # config = {
    #     'client': {
    #         'learning_rate': 0.001,
    #         'batch_size': 32
    #     }
    # }
    #
    # train_dataset = DummyDataset()
    # test_dataset = DummyDataset()
    # model = DummyModel()
    #
    # client = Client(client_id=1, model=model, train_dataset=train_dataset, test_dataset=test_dataset, config=config)
    #
    # print(f"Client {client.client_id} training...")
    # train_loss = client.train()
    # print(f"Client {client.client_id} train loss: {train_loss:.4f}")
    #
    # print(f"Client {client.client_id} testing...")
    # accuracy, _, _ = client.test()
    # print(f"Client {client.client_id} test accuracy: {accuracy:.4f}")
    #
    # print(f"Client {client.client_id} simulating label flipping attack...")
    # client.simulate_attack('label_flipping', {'flip_ratio': 0.2})
    # print("Attack simulated.")
    #
    # print(f"Client {client.client_id} simulating Gaussian noise attack...")
    # client.simulate_attack('gaussian_noise', {'noise_std': 0.05, 'target': 'weights'})
    # print("Attack simulated.")