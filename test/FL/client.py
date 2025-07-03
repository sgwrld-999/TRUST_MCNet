"""
Federated Learning Client Implementation

This module implements the client-side logic for federated learning using the Flower framework.
Each client represents a participant in the federated learning process that trains a model
locally on its private data and participates in the global model aggregation.

The client implements the core FL workflow:
1. Receives global model parameters from the server
2. Trains the model locally on its private dataset
3. Sends updated model parameters back to the server
4. Evaluates model performance on local validation data

This design preserves data privacy as raw data never leaves the client's environment,
only model parameters are shared with the central server.

Key Features:
- Local model training with configurable epochs and learning rate
- Model parameter serialization/deserialization for communication
- Local validation for performance monitoring
- GPU support for accelerated training
- Integration with PyTorch optimizers and loss functions

Author: [Your Name]
Date: [Current Date]
"""

import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any

class FlowerClient(fl.client.NumPyClient):
    """
    Federated Learning Client for Flower Framework
    
    This class implements a federated learning client that can participate in
    distributed model training while keeping data locally. It handles local
    training, model parameter exchange, and local evaluation.
    
    Why this class exists:
    - Encapsulates client-side federated learning logic
    - Provides interface for Flower server communication
    - Manages local training and evaluation processes
    - Ensures data privacy by keeping data local
    
    How it works:
    1. Receives global model parameters from FL server
    2. Updates local model with received parameters
    3. Performs local training on private data for specified epochs
    4. Returns updated parameters to server for aggregation
    5. Evaluates model performance on local validation set
    
    The client alternates between fit() and evaluate() calls orchestrated by the server.
    """
    
    def __init__(self, cfg, model, train_loader, val_loader):
        """
        Initialize the federated learning client.
        
        Parameters:
        cfg: Configuration object containing training hyperparameters
            - training.learning_rate: Learning rate for local optimizer
            - training.weight_decay: L2 regularization parameter
            - training.epochs: Number of local training epochs per FL round
            - ray.num_gpus: GPU availability for device selection
        model: PyTorch model instance to be trained
        train_loader: DataLoader for local training data partition
        val_loader: DataLoader for local validation data partition
        
        Use of parameters in simulation:
        - cfg: Controls local training behavior and device selection
        - model: The neural network architecture to train federatedly
        - train_loader: Provides local training data (simulates private data)
        - val_loader: Enables local performance evaluation
        """
        # Store configuration and initialize training components
        self.cfg = cfg
        
        # Device selection: Use GPU if available and configured, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.ray.num_gpus > 0 else "cpu")
        
        # Move model to selected device for training
        self.model = model.to(self.device)
        
        # Store data loaders for local training and validation
        self.train_loader = train_loader  # Private training data partition
        self.val_loader = val_loader      # Local validation data for performance monitoring
        
        # Initialize optimizer for local model training
        # SGD chosen for its stability in federated learning scenarios
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.training.learning_rate,    # Learning rate for gradient descent
            weight_decay=cfg.training.weight_decay  # L2 regularization to prevent overfitting
        )

    def get_parameters(self):
        """
        Extract model parameters for transmission to the FL server.
        
        This method converts PyTorch model parameters to NumPy arrays for
        network transmission. The Flower framework requires parameters in
        NumPy format for efficient serialization and communication.
        
        Why this is needed:
        - Enables parameter sharing between client and server
        - Required by Flower's communication protocol
        - Converts between PyTorch tensors and serializable NumPy arrays
        
        How it works:
        - Iterates through model's state dictionary
        - Converts each parameter tensor to CPU (if on GPU)
        - Converts CPU tensors to NumPy arrays
        
        Returns:
        List[np.ndarray]: Model parameters as NumPy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Update local model with parameters received from the FL server.
        
        This method updates the client's local model with the global model
        parameters received from the server. This typically happens at the
        beginning of each FL round when clients receive the latest global model.
        
        Why this is needed:
        - Synchronizes local model with global model state
        - Essential for federated learning parameter aggregation
        - Ensures all clients start each round with the same global model
        
        How it works:
        - Maps received NumPy arrays to model parameter names
        - Converts NumPy arrays back to PyTorch tensors
        - Loads the new parameters into the model's state dictionary
        
        Parameters:
        parameters (List[np.ndarray]): Model parameters from server as NumPy arrays
        
        Use in simulation:
        - Called before each local training round
        - Ensures model consistency across all federated clients
        - Implements the "broadcast" phase of federated learning
        """
        # Map received parameters to model parameter names and convert to tensors
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config) -> tuple:
        """
        Perform local model training on the client's private data.
        
        This method implements the core local training phase of federated learning.
        It updates the model with global parameters, trains locally for specified
        epochs, and returns the updated parameters for server aggregation.
        
        Why this is needed:
        - Implements the "local training" phase of federated learning
        - Enables clients to improve the model using their private data
        - Maintains data privacy by keeping data local during training
        
        How it works:
        1. Update local model with global parameters from server
        2. Set model to training mode for gradient computation
        3. Iterate through local training epochs
        4. For each batch: forward pass, loss computation, backpropagation
        5. Return updated parameters and training metadata
        
        Parameters:
        parameters: Global model parameters from server
        config: Training configuration (unused in this implementation)
        
        Returns:
        tuple: (updated_parameters, num_training_samples, metrics_dict)
            - updated_parameters: Locally trained model parameters
            - num_training_samples: Size of local training dataset
            - metrics_dict: Training metrics (empty in this implementation)
        
        Use in simulation:
        - Called once per FL round for each client
        - Simulates private local training on distributed data
        - Provides updated parameters for global model aggregation
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Set model to training mode (enables dropout, batch norm training mode)
        self.model.train()
        
        # Local training loop for specified number of epochs
        for _ in range(self.cfg.training.epochs):
            for data, target in self.train_loader:
                # Move data to appropriate device (CPU/GPU)
                data, target = data.to(self.device), target.to(self.device)
                
                # Reset gradients from previous iteration
                self.optimizer.zero_grad()
                
                # Forward pass: compute model predictions
                output = self.model(data)
                
                # Compute loss using negative log likelihood (for classification)
                loss = F.nll_loss(output, target)
                
                # Backward pass: compute gradients
                loss.backward()
                
                # Update model parameters using computed gradients
                self.optimizer.step()
                
        # Return updated parameters and training metadata
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config) -> tuple:
        """
        Evaluate the model on local validation data.
        
        This method evaluates the current model performance on the client's
        local validation set. It provides insights into model performance
        without sharing private validation data with the server.
        
        Why this is needed:
        - Monitors local model performance during federated learning
        - Provides validation metrics without data sharing
        - Helps assess model quality and convergence
        - Enables early stopping and hyperparameter tuning
        
        How it works:
        1. Update local model with latest global parameters
        2. Set model to evaluation mode (disables dropout, batch norm changes)
        3. Iterate through validation data without gradient computation
        4. Compute loss and accuracy metrics
        5. Return performance metrics and validation set size
        
        Parameters:
        parameters: Current global model parameters
        config: Evaluation configuration (unused in this implementation)
        
        Returns:
        tuple: (loss, num_validation_samples, metrics_dict)
            - loss: Average validation loss
            - num_validation_samples: Size of validation dataset
            - metrics_dict: Additional metrics (accuracy in this case)
        
        Use in simulation:
        - Called after each FL round for performance monitoring
        - Provides distributed validation without data centralization
        - Helps track model improvement across FL rounds
        """
        # Update local model with current global parameters
        self.set_parameters(parameters)
        
        # Set model to evaluation mode (disables dropout, fixes batch norm)
        self.model.eval()
        
        # Initialize metrics tracking
        loss, correct = 0.0, 0
        
        # Evaluate on validation data without gradient computation
        for data, target in self.val_loader:
            # Move data to appropriate device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass without gradient tracking
            output = self.model(data)
            
            # Accumulate loss (sum over all validation samples)
            loss += F.nll_loss(output, target, reduction="sum").item()
            
            # Count correct predictions for accuracy calculation
            pred = output.argmax(dim=1)  # Get predicted class
            correct += (pred == target).sum().item()
        
        # Calculate average loss and accuracy
        loss /= len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        
        # Return evaluation metrics
        return float(loss), len(self.val_loader.dataset), {"accuracy": accuracy}


def make_client(cfg, model_cls, train_loader, val_loader):
    """
    Factory function to create a configured FlowerClient instance.
    
    This function serves as a factory for creating federated learning clients
    with the specified configuration and data partitions. It instantiates the
    model and creates a client ready for federated learning participation.
    
    Why this function exists:
    - Provides a clean interface for client creation
    - Encapsulates model instantiation logic
    - Simplifies client setup in the main simulation loop
    - Enables easy testing and experimentation with different configurations
    
    How it works:
    - Instantiates the specified model class
    - Creates a FlowerClient with the model and data loaders
    - Returns the configured client ready for FL participation
    
    Parameters:
    cfg: Configuration object with training and system parameters
    model_cls: Model class to instantiate (e.g., Net)
    train_loader: DataLoader for client's training data partition
    val_loader: DataLoader for client's validation data partition
    
    Returns:
    FlowerClient: Configured client instance ready for federated learning
    
    Use in simulation:
    - Called by the main simulation loop for each client
    - Provides consistent client initialization across all participants
    - Enables easy swapping of model architectures and configurations
    """
    # Instantiate the model class
    model = model_cls()
    
    # Create and return configured FlowerClient
    return FlowerClient(cfg, model, train_loader, val_loader)