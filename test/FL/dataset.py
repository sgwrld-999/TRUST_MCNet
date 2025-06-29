import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist(data_path: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return train_set, test_set

def partition_dataset(train_set, num_clients):
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(train_set) - sum(lengths)
    lengths[-1] += remainder
    return random_split(train_set, lengths, generator=torch.Generator().manual_seed(2023))

def split_train_val(partition, val_ratio=0.1):
    num_total = len(partition)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val
    return random_split(partition, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

def create_loaders(partitions, batch_size=20, val_ratio=0.1):
    train_loaders, val_loaders = [], []
    for partition in partitions:
        train_part, val_part = split_train_val(partition, val_ratio)
        train_loader = DataLoader(train_part, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_part, batch_size=batch_size, shuffle=False, num_workers=2)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return train_loaders, val_loaders

def create_test_loader(test_set, batch_size=128):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)

def partition_dirichlet(train_dataset, num_clients, alpha=0.5):
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx) < len(labels) / num_clients) for p, idx in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)
        for idx, client_idx in zip(split_indices, client_indices):
            client_idx.extend(idx.tolist())
    client_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]
    return client_datasets

def prepare_datasets(num_clients=10, batch_size=20, val_ratio=0.1,
                     partitioning="iid", alpha=0.5):
    train_set, test_set = get_mnist()
    if partitioning == "iid":
        partitions = partition_dataset(train_set, num_clients)
    elif partitioning == "dirichlet":
        partitions = partition_dirichlet(train_set, num_clients, alpha)
    else:
        raise ValueError("Unknown partitioning method")
    train_loaders, val_loaders = create_loaders(partitions, batch_size, val_ratio)
    test_loader = create_test_loader(test_set)
    return train_loaders, val_loaders, test_loader
