import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Client:
    def __init__(self, client_id, model, train_dataset, test_dataset, config):
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['client']['learning_rate'])
        self.criterion = nn.CrossEntropyLoss() # Assuming classification for anomaly detection

    def train(self):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.config['client']['batch_size'], shuffle=True)
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def test(self):
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=self.config['client']['batch_size'], shuffle=False)
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = correct / total
        return accuracy, all_labels, all_predictions

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def simulate_attack(self, attack_type, attack_params):
        """
        Simulates different types of attacks on the client's data or model updates.
        Args:
            attack_type (str): Type of attack ('label_flipping', 'gaussian_noise', etc.)
            attack_params (dict): Dictionary of parameters specific to the attack.
        """
        if attack_type == 'label_flipping':
            self._apply_label_flipping(attack_params)
        elif attack_type == 'gaussian_noise':
             self._apply_gaussian_noise(attack_params)
        # Add more attack types as needed
        else:
            print(f"Warning: Unknown attack type '{attack_type}' for client {self.client_id}.")

    def _apply_label_flipping(self, params):
        """
        Applies label flipping to a percentage of the training data.
        Flips anomaly labels to normal and vice versa.
        Assumes binary classification (0: normal, 1: anomaly).
        """
        flip_ratio = params.get('flip_ratio', 0.1) # Percentage of labels to flip

        if not hasattr(self.train_dataset, '__len__') or not hasattr(self.train_dataset, '__getitem__'):
            print(f"Warning: Client {self.client_id}'s train_dataset does not support label flipping.")
            return

        num_samples_to_flip = int(len(self.train_dataset) * flip_ratio)
        indices_to_flip = np.random.choice(len(self.train_dataset), num_samples_to_flip, replace=False)

        # Assuming the dataset returns (data, label) tuples
        if hasattr(self.train_dataset, 'targets'):
            for i in indices_to_flip:
                # Assuming targets is a list or numpy array of labels
                self.train_dataset.targets[i] = 1 - self.train_dataset.targets[i]
        else:
            print(f"Warning: Client {self.client_id}'s train_dataset structure is not compatible with label flipping.")


    def _apply_gaussian_noise(self, params):
        """
        Adds Gaussian noise to the model's gradients or weights after training.
        """
        noise_std = params.get('noise_std', 0.01) # Standard deviation of Gaussian noise
        noise_target = params.get('target', 'gradients') # 'gradients' or 'weights'

        if noise_target == 'gradients':
            # Add noise to gradients after calculating them
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_std
                    param.grad.data.add_(noise)
        elif noise_target == 'weights':
             # Add noise to weights after training (before sending to server)
             with torch.no_grad():
                 for param in self.model.parameters():
                     noise = torch.randn_like(param.data) * noise_std
                     param.data.add_(noise)
        else:
            print(f"Warning: Invalid noise target '{noise_target}' for client {self.client_id}.")


if __name__ == '__main__':
    # Example Usage (requires dummy dataset and model)
    print("Client class defined. Example usage requires defining dummy dataset and model.")

    # class DummyDataset(Dataset):
    #     def __init__(self, num_samples=100, input_dim=10, output_dim=2):
    #         self.data = torch.randn(num_samples, input_dim)
    #         self.targets = torch.randint(0, output_dim, (num_samples,))
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         return self.data[idx], self.targets[idx]
    #
    # class DummyModel(nn.Module):
    #     def __init__(self, input_dim=10, output_dim=2):
    #         super(DummyModel, self).__init__()
    #         self.fc = nn.Linear(input_dim, output_dim)
    #
    #     def forward(self, x):
    #         return self.fc(x)
    #
    # config = {
    #     'client': {
    #         'learning_rate': 0.001,
    #         'batch_size': 32
    #     }
    # }
    #
    # train_dataset = DummyDataset()
    # test_dataset = DummyDataset()
    # model = DummyModel()
    #
    # client = Client(client_id=1, model=model, train_dataset=train_dataset, test_dataset=test_dataset, config=config)
    #
    # print(f"Client {client.client_id} training...")
    # train_loss = client.train()
    # print(f"Client {client.client_id} train loss: {train_loss:.4f}")
    #
    # print(f"Client {client.client_id} testing...")
    # accuracy, _, _ = client.test()
    # print(f"Client {client.client_id} test accuracy: {accuracy:.4f}")
    #
    # print(f"Client {client.client_id} simulating label flipping attack...")
    # client.simulate_attack('label_flipping', {'flip_ratio': 0.2})
    # print("Attack simulated.")
    #
    # print(f"Client {client.client_id} simulating Gaussian noise attack...")
    # client.simulate_attack('gaussian_noise', {'noise_std': 0.05, 'target': 'weights'})
    # print("Attack simulated.")