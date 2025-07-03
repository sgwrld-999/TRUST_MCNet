"""
Federated Learning Dataset Management (Duplicate)

NOTE: This file appears to be a duplicate of dataset.py and contains identical
DataManager class implementation. This may be a result of development iteration
or backup purposes.

For the main implementation and documentation, please refer to dataset.py.
This file should be removed or repurposed to avoid confusion and maintain
clean codebase organization.

If this file serves a specific purpose different from dataset.py, please:
1. Rename it to reflect its specific purpose
2. Add appropriate documentation explaining its unique functionality
3. Remove duplicate code and import from dataset.py instead

Author: [Your Name]
Date: [Current Date]
"""

# TODO: Remove this duplicate file or clarify its specific purpose
# Main implementation is in dataset.py

import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

class DataManager:
    def __init__(self, cfg):
        self.data_path = cfg.dataset.data_path
        self.batch_size = cfg.dataset.batch_size
        self.val_ratio = cfg.dataset.val_ratio
        self.partitioning = cfg.dataset.partitioning
        self.alpha = cfg.dataset.dirichlet_alpha

    def get_mnist(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(self.data_path, train=True, download=True, transform=transform)
        test = datasets.MNIST(self.data_path, train=False, download=True, transform=transform)
        return train, test

    def partition(self, train_set, num_clients):
        if self.partitioning == "iid":
            lengths = [len(train_set) // num_clients] * num_clients
            lengths[-1] += len(train_set) - sum(lengths)
            splits = random_split(train_set, lengths)
        elif self.partitioning == "dirichlet":
            data_indices = [[] for _ in range(num_clients)]
            targets = np.array(train_set.targets)
            num_classes = len(np.unique(targets))
            for c in range(num_classes):
                idx_k = np.where(targets == c)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                splits_k = np.split(idx_k, split_points)
                for client_id, idx in enumerate(splits_k):
                    data_indices[client_id] += idx.tolist()
            splits = [Subset(train_set, inds) for inds in data_indices]
        else:
            raise ValueError(f"Unknown partitioning: {self.partitioning}")
        return splits

    def create_loaders(self, splits):
        train_loaders, val_loaders = [], []
        for subset in splits:
            val_size = int(len(subset) * self.val_ratio)
            train_size = len(subset) - val_size
            train_sub, val_sub = random_split(subset, [train_size, val_size])
            train_loaders.append(DataLoader(train_sub, batch_size=self.batch_size, shuffle=True))
            val_loaders.append(DataLoader(val_sub, batch_size=self.batch_size, shuffle=False))
        return train_loaders, val_loaders

    def create_test_loader(self, test_set):
        return DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

    def prepare(self, num_clients):
        train_set, test_set = self.get_mnist()
        splits = self.partition(train_set, num_clients)
        train_loaders, val_loaders = self.create_loaders(splits)
        test_loader = self.create_test_loader(test_set)
        return train_loaders, val_loaders, test_loader