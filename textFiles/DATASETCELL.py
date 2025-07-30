# DATASET CELL
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

class ToNIoTDataset:
    """
    Simulated ToN-IoT dataset for IoT Network Intrusion Detection
    Creates partitioned data for federated learning clients
    """
    def __init__(self, partition_id, total_partitions, config):
        # Set seed for reproducibility but vary by client ID
        np.random.seed(42 + partition_id)

        # Dataset parameters
        n_samples = 6000
        n_features = 42  # Common in IoT network flow feature extraction
        
        # Create class imbalance: 80% normal, 20% anomalies
        # Normal traffic (label 0) - centered around 0 with small variance
        normal = np.random.normal(0, 1, (int(0.8*n_samples), n_features))
        
        # Anomalous traffic (label 1) - different distribution
        anomaly = np.random.normal(2, 1.5, (int(0.2*n_samples), n_features))
        
        # Combine data and create labels
        X = np.vstack((normal, anomaly))
        y = np.hstack((np.zeros(normal.shape[0]), np.ones(anomaly.shape[0])))
        
        # Create non-IID partitioning by taking every nth sample
        # This ensures each client has a different data distribution
        X, y = X[partition_id::total_partitions], y[partition_id::total_partitions]
        
        # Optional: Add client-specific bias to simulate device heterogeneity
        client_bias = (partition_id - total_partitions/2) * 0.2
        X = X + client_bias
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Create PyTorch data loaders
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), 
            batch_size=config.batch_size
        )
        
        # Store dataset properties
        self.input_dim = X.shape[1]
        self.num_classes = 2
        self.partition_id = partition_id
        self.n_train_samples = len(y_train)
        self.n_test_samples = len(y_test)
        self.anomaly_ratio = np.mean(y)
        
    def get_dataset_stats(self):
        """Return key statistics about this client's dataset partition"""
        return {
            'client_id': self.partition_id,
            'train_samples': self.n_train_samples,
            'test_samples': self.n_test_samples,
            'anomaly_ratio': self.anomaly_ratio,
            'input_dimension': self.input_dim
        }

# Example usage (to visualize)
if __name__ == "__main__":
    from types import SimpleNamespace
    sample_config = SimpleNamespace(batch_size=64)
    
    # Create 3 client datasets
    datasets = [ToNIoTDataset(i, 3, sample_config) for i in range(3)]
    
    # Print statistics for each client's data
    for i, dataset in enumerate(datasets):
        stats = dataset.get_dataset_stats()
        print(f"Client {i} stats:", stats)