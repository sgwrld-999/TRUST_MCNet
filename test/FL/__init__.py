"""
Federated Learning Simulation Package

This package implements a complete federated learning simulation framework using
the Flower federated learning library and PyTorch for deep learning. The package
provides all necessary components to simulate realistic federated learning scenarios
with configurable parameters and data distributions.

Package Components:
==================

main.py - Entry Point and Orchestration
---------------------------------------
- Main simulation coordinator using Hydra configuration management
- Ray integration for distributed client processing
- Flower simulation setup and execution
- Resource management and cleanup

client.py - Federated Learning Client Implementation
---------------------------------------------------
- FlowerClient class implementing local training and evaluation
- Parameter serialization/deserialization for server communication
- Local model training with configurable epochs and learning rates
- Local validation for performance monitoring
- GPU/CPU device management

dataset.py - Data Management and Partitioning
---------------------------------------------
- DataManager class for MNIST dataset preparation
- IID and non-IID (Dirichlet) data partitioning strategies
- Train/validation splitting for each client
- DataLoader creation for efficient batch processing
- Configurable preprocessing and normalization

model.py - Neural Network Architecture
--------------------------------------
- Simple linear classifier optimized for federated learning
- MNIST-specific architecture (784 → 10 linear layer)
- Log-softmax activation for numerical stability
- Minimal parameters for efficient FL communication

server.py - Federated Learning Server
-------------------------------------
- Flower server implementation for FL coordination
- Client management and round orchestration
- FedAvg parameter aggregation
- Network communication and fault tolerance

Configuration:
=============
The simulation is configured through conf/config.yaml with parameters for:
- Dataset settings (batch size, partitioning strategy, validation ratio)
- Training hyperparameters (learning rate, epochs, number of clients/rounds)
- System resources (Ray CPU/GPU allocation, memory management)
- Server networking (address, port configuration)

Usage:
======
Run the federated learning simulation with:
    python main.py

The simulation will:
1. Initialize Ray for distributed processing
2. Partition MNIST data across specified number of clients
3. Create federated learning clients with local data
4. Execute specified number of FL rounds with FedAvg aggregation
5. Report training progress and final performance metrics

Key Features:
============
- Configurable IID/non-IID data distributions
- Scalable client simulation using Ray
- GPU acceleration support
- Comprehensive logging and metrics
- Modular design for easy experimentation
- Production-ready Flower integration

Research Applications:
=====================
This package is designed for federated learning research including:
- Algorithm development and evaluation
- Data heterogeneity studies
- System scalability testing
- Privacy-preserving machine learning experiments
- Benchmarking against centralized baselines

The simple architecture and comprehensive configuration options make it ideal
for both educational purposes and serious federated learning research.

Author: [Your Name]
Date: [Current Date]
Version: 1.0
License: [Your License]
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[Your Email]"
__description__ = "Federated Learning Simulation Framework with Flower and PyTorch"

# Main package imports for easy access
from .main import main
from .client import FlowerClient, make_client
from .dataset import DataManager
from .model import Net

# Define public API
__all__ = [
    'main',
    'FlowerClient', 
    'make_client',
    'DataManager',
    'Net'
]