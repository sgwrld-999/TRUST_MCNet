# MODEL CELL
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    """
    Neural network for IoT anomaly detection
    
    Architecture:
    - Input layer: Takes IoT network features
    - Two hidden layers with ReLU activations
    - Output layer: Binary classification (normal/anomaly)
    """
    def __init__(self, input_dim, num_classes, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Output layer
        self.out = nn.Linear(hidden_dim2, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward(self, x):
        """Forward pass through the network"""
        # First layer with ReLU
        x = F.relu(self.fc1(x))
        # Second layer with ReLU
        x = F.relu(self.fc2(x))
        # Output logits (no activation - will use CrossEntropyLoss)
        return self.out(x)
    
    def get_parameter_count(self):
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage (for visualization)
if __name__ == "__main__":
    # Create a sample model
    model = SimpleModel(input_dim=42, num_classes=2)
    # Print model architecture
    print(model)
    # Print parameter count
    print(f"Trainable parameters: {model.get_parameter_count():,}")