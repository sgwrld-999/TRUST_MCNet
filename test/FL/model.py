"""
Neural Network Model for Federated Learning

This module defines the neural network architecture used in the federated learning
simulation. It implements a simple linear classifier designed specifically for
MNIST digit classification in a federated learning context.

The model architecture is intentionally simple to:
1. Focus on federated learning dynamics rather than model complexity
2. Enable fast training and convergence in FL scenarios
3. Provide interpretable results for FL research and experimentation
4. Minimize computational requirements for distributed training

Key Design Decisions:
- Single linear layer: Simplifies federated aggregation and reduces communication
- Log-softmax activation: Provides numerical stability for classification
- Flattened input: Handles MNIST's 28x28 pixel images efficiently
- 10 output classes: Matches MNIST's digit classification task

This simple architecture is ideal for federated learning experiments as it:
- Trains quickly on distributed data partitions
- Requires minimal parameter communication between clients and server
- Provides clear baselines for FL algorithm evaluation
- Enables focus on data heterogeneity effects rather than model complexity

Author: [Your Name]
Date: [Current Date]
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Simple Linear Neural Network for MNIST Classification in Federated Learning
    
    This class implements a basic linear classifier designed specifically for
    federated learning experiments on MNIST digit classification. The architecture
    prioritizes simplicity and efficiency over complexity to enable clear analysis
    of federated learning dynamics.
    
    Why this simple architecture:
    - Fast convergence: Enables quick FL experiments and prototyping
    - Low communication cost: Fewer parameters to transmit between clients/server
    - Clear baselines: Provides interpretable results for FL research
    - Computational efficiency: Suitable for resource-constrained FL environments
    - Focus on FL dynamics: Removes model complexity as a confounding factor
    
    How the model works:
    1. Input: 28x28 MNIST images flattened to 784-dimensional vectors
    2. Linear transformation: Maps input features to 10 class logits
    3. Log-softmax: Converts logits to log-probabilities for stable training
    4. Output: 10-dimensional log-probability distribution over digit classes
    
    Architecture Details:
    - Single fully connected layer (784 → 10)
    - No hidden layers or non-linear activations
    - Log-softmax output for numerical stability
    - Compatible with NLL (Negative Log Likelihood) loss
    
    This architecture serves as an excellent baseline for federated learning
    research, allowing researchers to focus on FL algorithms, data heterogeneity
    effects, and system properties rather than model architecture optimization.
    """
    
    def __init__(self, input_dim=28*28, num_classes=10):
        """
        Initialize the linear neural network.
        
        This constructor sets up a simple linear classifier with configurable
        input dimensions and output classes, though defaults are optimized
        for MNIST digit classification.
        
        Parameters:
        input_dim (int, default=784): Input feature dimension
            - For MNIST: 28x28 = 784 pixels per image
            - Flattened from 2D image to 1D feature vector
            - Can be adjusted for other datasets or preprocessing
        
        num_classes (int, default=10): Number of output classes
            - For MNIST: 10 digit classes (0-9)
            - Output layer will have this many neurons
            - Determines size of final classification layer
        
        Use of parameters in simulation:
        - input_dim: Must match flattened MNIST image size (784)
        - num_classes: Must match MNIST digit classes (10)
        - These parameters enable easy adaptation to other datasets
        - Default values are optimized for standard MNIST FL experiments
        """
        super().__init__()
        
        # Single fully connected layer for linear classification
        # Maps flattened input images directly to class logits
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the neural network.
        
        This method defines the forward computation of the model, transforming
        input MNIST images into class predictions through a simple linear
        transformation followed by log-softmax activation.
        
        Why this forward pass design:
        - Flattening: Converts 2D images to 1D vectors for linear layer processing
        - Linear transformation: Learns direct mapping from pixels to class logits
        - Log-softmax: Provides numerically stable probability distributions
        - Simplicity: Enables focus on FL dynamics rather than model complexity
        
        How it works:
        1. Flatten input: Reshape (batch_size, 28, 28) → (batch_size, 784)
        2. Linear transformation: Apply learned weights and biases
        3. Log-softmax: Convert logits to log-probabilities for stable training
        4. Output: Log-probability distribution over 10 digit classes
        
        Mathematical operation:
        log_probs = log_softmax(W * flatten(x) + b)
        where W is the weight matrix and b is the bias vector
        
        Parameters:
        x (torch.Tensor): Input batch of MNIST images
            - Shape: (batch_size, 1, 28, 28) for MNIST
            - Values: Normalized pixel intensities [0, 1]
            - Batch dimension allows processing multiple images simultaneously
        
        Returns:
        torch.Tensor: Log-probability predictions for each class
            - Shape: (batch_size, 10) for MNIST
            - Values: Log-probabilities summing to 1 per image
            - Compatible with NLL loss for training
        
        Use in federated learning simulation:
        - Called during local training on each client's data partition
        - Used for local validation and performance evaluation
        - Provides predictions for loss computation and accuracy metrics
        - Simple architecture enables efficient parameter communication in FL
        """
        # Flatten input images from 2D to 1D for linear layer processing
        # Shape: (batch_size, 1, 28, 28) → (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # Apply linear transformation and log-softmax activation
        # Returns log-probabilities for stable numerical computation
        return F.log_softmax(self.fc(x), dim=1)