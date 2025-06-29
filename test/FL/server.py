import flwr as fl

def main():
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5)
    )

if __name__ == "__main__":
    main()

