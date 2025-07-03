"""
Federated Learning Simulation Entry Point

This module serves as the main entry point for running federated learning simulations
using the Flower framework. It orchestrates the entire federated learning process by
coordinating multiple clients and a central server.

The simulation mimics a real-world federated learning scenario where multiple clients
(representing different devices or organizations) collaboratively train a machine learning
model while keeping their data locally distributed and private.

Key Components:
- Ray: Distributed computing framework for parallelizing client operations
- Flower: Federated learning framework for managing FL orchestration
- Hydra: Configuration management for experiment parameters
- Custom components: DataManager, Neural Network Model, and Client implementations

Author: [Your Name]
Date: [Current Date]
"""

import hydra
from omegaconf import DictConfig
import ray
import flwr as fl
from flwr.server import ServerConfig
from dataset import DataManager
from model import Net
from client import make_client

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function that orchestrates the federated learning simulation.
    
    This function coordinates the entire federated learning process by:
    1. Initializing Ray for distributed computing
    2. Preparing federated datasets for multiple clients
    3. Creating client instances with their respective data partitions
    4. Running the federated learning simulation with specified rounds
    5. Cleaning up resources after simulation completion
    
    Why this approach:
    - Simulates real-world federated learning where data is distributed across clients
    - Uses Ray for efficient parallel processing of multiple clients
    - Leverages Flower's built-in aggregation algorithms (FedAvg by default)
    - Provides configurable parameters for experimentation
    
    How it works:
    1. Ray initialization creates a distributed computing environment
    2. DataManager partitions the dataset across multiple clients (IID or non-IID)
    3. Each client gets its own data partition (training and validation)
    4. Flower orchestrates the federated learning rounds:
       - Clients train locally on their data
       - Server aggregates model parameters
       - Updated global model is sent back to clients
    5. Process repeats for the specified number of rounds
    
    Parameters:
    cfg (DictConfig): Hydra configuration object containing:
        - ray.num_cpus: Number of CPU cores for Ray (for parallel client processing)
        - ray.num_gpus: Number of GPUs for Ray (for GPU-accelerated training)
        - ray.object_store_memory: Memory allocation for Ray's object store
        - training.num_clients: Number of federated learning clients to simulate
        - training.num_rounds: Number of federated learning rounds to execute
        
    Use of parameters in simulation:
    - num_cpus/num_gpus: Control computational resources for parallel client training
    - object_store_memory: Manages memory for storing model parameters during aggregation
    - num_clients: Determines data partitioning and simulation scale
    - num_rounds: Controls the number of federated learning iterations
    
    Returns:
    None: The function orchestrates the simulation and prints results
    """
    # Initialize Ray distributed computing framework
    # Ray enables parallel processing of multiple FL clients simultaneously
    ray.init(
        num_cpus=cfg.ray.num_cpus,          # CPU cores for parallel client execution
        num_gpus=cfg.ray.num_gpus,          # GPUs for accelerated model training
        object_store_memory=cfg.ray.object_store_memory  # Memory for storing model parameters
    )

    # Initialize data manager and prepare federated datasets
    # DataManager handles dataset partitioning across clients (IID or non-IID distribution)
    dm = DataManager(cfg)
    train_loaders, val_loaders, test_loader = dm.prepare(cfg.training.num_clients)

    def client_fn(cid: str):
        """
        Client factory function that creates FL client instances.
        
        This function is called by Flower for each client in the simulation.
        Each client gets its own data partition and model instance.
        
        Parameters:
        cid (str): Client identifier (string representation of client index)
        
        Returns:
        FlowerClient: Configured client instance with assigned data partition
        """
        idx = int(cid)  # Convert client ID to integer index
        return make_client(cfg, Net, train_loaders[idx], val_loaders[idx])

    # Start federated learning simulation
    # Flower orchestrates the FL process: local training + global aggregation
    fl.simulation.start_simulation(
        client_fn=client_fn,                              # Function to create client instances
        num_clients=cfg.training.num_clients,             # Total number of clients in simulation
        config=ServerConfig(num_rounds=cfg.training.num_rounds),  # FL rounds configuration
    )

    # Clean up Ray resources after simulation completion
    ray.shutdown()

if __name__ == "__main__":
    main()
