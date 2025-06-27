# TRUST MCNet (Multi-Client Network)

A federated learning framework for anomaly detection with trust mechanisms and multi-client coordination.

## Project Overview

TRUST MCNet is a distributed machine learning system that enables multiple clients to collaboratively train models while maintaining data privacy. The framework includes trust mechanisms to ensure reliable collaboration between clients.

## Features

- **Federated Learning**: Distributed training across multiple clients
- **Trust Mechanisms**: Built-in trust evaluation for client reliability
- **Multi-Model Support**: Supports both MLP and LSTM architectures
- **Anomaly Detection**: Specialized for anomaly detection tasks
- **Privacy Preservation**: Client data remains local during training

## Project Structure

```
TRUST_MCNet/
├── TRUST_MCNet_Codebase/
│   ├── clients/
│   │   └── client.py           # Client implementation for federated learning
│   ├── config/
│   │   └── config.yaml         # Configuration settings
│   ├── data/
│   │   ├── data_loader.py      # Data loading and preprocessing utilities
│   │   └── preprocess.py       # Data preprocessing functions
│   └── models/
│       └── model.py            # Neural network models (MLP, LSTM)
├── Data_Pipeline_TRUST_MCNet.png  # System architecture diagram
├── Problem_stat2206.pdf           # Problem statement document
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- PyYAML

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TRUST_MCNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `TRUST_MCNet_Codebase/config/config.yaml` to configure:
- Model parameters
- Training settings
- Client configurations
- Data preprocessing options

## Usage

### Basic Usage

1. Configure your settings in `config/config.yaml`
2. Prepare your data using the data preprocessing utilities
3. Initialize clients and start federated training

```python
from TRUST_MCNet_Codebase.clients.client import Client
from TRUST_MCNet_Codebase.models.model import MLP, LSTM
from TRUST_MCNet_Codebase.data.data_loader import ConfigManager

# Load configuration
config = ConfigManager("TRUST_MCNet_Codebase/config/config.yaml")

# Initialize model and client
model = MLP(input_dim=10, hidden_dims=[64, 32], output_dim=2)
client = Client(client_id=1, model=model, train_dataset=train_data, 
                test_dataset=test_data, config=config.cfg)

# Train the client
loss = client.train()
accuracy = client.test()
```

## Models

### MLP (Multi-Layer Perceptron)
- Configurable hidden dimensions
- ReLU activation functions
- Suitable for tabular data

### LSTM (Long Short-Term Memory)
- Configurable hidden dimensions and layers
- Designed for sequential data
- Supports variable sequence lengths

## Client Architecture

Each client in the federated learning setup:
- Maintains local data privacy
- Performs local model training
- Participates in global model aggregation
- Includes trust evaluation mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the repository.

---

**Note**: This is a research project focused on federated learning and trust mechanisms in distributed machine learning systems.
