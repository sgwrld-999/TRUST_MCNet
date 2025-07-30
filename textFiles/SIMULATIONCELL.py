# SIMULATION CELL
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import start_simulation
from flwr.client import ClientApp
from flwr.common import Context
import matplotlib.pyplot as plt
import pandas as pd

def client_fn(context: Context):
    """Create a client instance based on the context"""
    # Get the client's partition ID from the context
    partition_id = int(context.node_config["partition-id"])
    
    # Create the client dataset
    dataset = ToNIoTDataset(partition_id, config.num_clients, config)
    
    # Print dataset stats
    print(f"Client {partition_id} dataset stats: {dataset.get_dataset_stats()}")
    
    # Create and initialize the model
    model = SimpleModel(
        input_dim=dataset.input_dim, 
        num_classes=dataset.num_classes,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2
    )
    
    # Create and return the client
    return TrustMCClient(
        model=model, 
        train_loader=dataset.train_loader, 
        test_loader=dataset.test_loader,
        client_id=f"client_{partition_id}"
    ).to_client()

# Create the client app
client_app = ClientApp(client_fn)

def server_fn(context):
    """Create a server instance with the DTWA strategy"""
    # Initialize the TrustMCStrategy
    strategy = TrustMCStrategy(
        # Trust-related parameters
        percentile=config.percentile_threshold,
        temp0=config.temperature,
        # Other strategy parameters
        min_fit_clients=config.num_clients,
        min_available_clients=config.num_clients
    )
    
    # Return server components with configured strategy
    return ServerAppComponents(
        strategy=strategy, 
        config=ServerConfig(num_rounds=config.num_rounds)
    )

# Create the server app
server_app = ServerApp(server_fn)

# Run the federated simulation
print(f"Starting federated simulation with {config.num_clients} clients for {config.num_rounds} rounds")

# Use start_simulation instead of run_simulation for better compatibility
history = start_simulation(
    client_fn=client_fn,
    num_clients=config.num_clients,
    config=ServerConfig(num_rounds=config.num_rounds),
    strategy=TrustMCStrategy(
        percentile=config.percentile_threshold,
        temp0=config.temperature,
        min_fit_clients=config.num_clients,
        min_available_clients=config.num_clients
    ),
    client_resources={"num_cpus": 1},
    ray_init_args={"ignore_reinit_error": True, "include_dashboard": False}
)

print("Simulation complete!")

# Display comprehensive simulation results
print("\n" + "="*50)
print("SIMULATION RESULTS SUMMARY")
print("="*50)

# Check for distributed metrics (from client evaluations)
if hasattr(history, 'metrics_distributed') and len(history.metrics_distributed) > 0:
    print(f"✅ Distributed evaluation rounds completed: {len(history.metrics_distributed)}")
    
    # Get final round metrics from all clients
    final_round_metrics = history.metrics_distributed[-1]
    if final_round_metrics:
        # Calculate average metrics across all clients
        accuracies = [metrics['accuracy'] for _, metrics in final_round_metrics if 'accuracy' in metrics]
        losses = [metrics['loss'] for _, metrics in final_round_metrics if 'loss' in metrics]
        
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"Final Average Client Accuracy: {avg_accuracy:.4f}")
            print(f"Client Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
        
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"Final Average Client Loss: {avg_loss:.4f}")
            
        print(f"Number of participating clients: {len(final_round_metrics)}")
else:
    print("No distributed metrics available")

# Check for distributed losses (from client training)
if hasattr(history, 'losses_distributed') and len(history.losses_distributed) > 0:
    print(f"Training rounds completed: {len(history.losses_distributed)}")
    final_losses = history.losses_distributed[-1]
    if final_losses:
        avg_train_loss = sum(loss for _, loss in final_losses) / len(final_losses)
        print(f"Final Average Training Loss: {avg_train_loss:.4f}")
else:
    print("No distributed training losses available")

# Check for centralized metrics
if hasattr(history, 'metrics_centralized') and len(history.metrics_centralized) > 0:
    print(f"Centralized evaluation: {history.metrics_centralized[-1]}")
else:
    print("No centralized evaluation (this is normal for client-only evaluation)")

print("="*50)