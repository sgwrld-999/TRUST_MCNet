import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader):
        self.model = Net().to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

    def train(self, epochs=1):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def test(self):
        self.model.eval()
        loss = 0
        correct = 0
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        return loss, accuracy

