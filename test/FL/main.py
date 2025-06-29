from dataset import prepare_datasets
from client import FlowerClient
import flwr as fl

NUM_CLIENTS = 5

train_loaders, val_loaders, test_loader = prepare_datasets(
    num_clients=NUM_CLIENTS, partitioning="dirichlet", alpha=0.5
)

def client_fn(cid: str):
    cid_int = int(cid)
    train_loader = train_loaders[cid_int]
    val_loader = val_loaders[cid_int]
    return FlowerClient(train_loader, val_loader)

if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=client_fn("0")
    )

