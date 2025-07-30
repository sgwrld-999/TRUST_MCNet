# CONFIG CELL
from omegaconf import OmegaConf
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    """Configuration parameters for the federated training process"""
    # Federated learning parameters
    num_rounds: int = 5           # Number of federated training rounds
    num_clients: int = 3          # Number of clients participating
    
    # Local training parameters
    batch_size: int = 64          # Batch size for local training
    learning_rate: float = 0.001  # Learning rate for client optimizers
    
    # Trust-related parameters
    trust_threshold: float = 0.5  # Minimum trust score for client inclusion
    percentile_threshold: float = 40.0  # Percentile for dynamic threshold
    temperature: float = 1.0      # Initial softmax temperature
    
    # Model parameters
    hidden_dim1: int = 128        # First hidden layer dimension
    hidden_dim2: int = 64         # Second hidden layer dimension

# Create configuration object
config = TrainingConfig()

# Convert to OmegaConf for easier manipulation
conf = OmegaConf.create(config.dict())
print("Configuration parameters:")
print(OmegaConf.to_yaml(conf))