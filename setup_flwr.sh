#!/bin/bash

# TRUST-MCNet Flwr Setup Script
# This script installs required dependencies and sets up the environment

echo "Setting up TRUST-MCNet with Flwr for IoT Federated Learning..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

echo "Installing required packages..."

# Install core dependencies
pip3 install torch>=1.9.0
pip3 install torchvision>=0.10.0
pip3 install scikit-learn>=1.0.0
pip3 install pandas>=1.3.0
pip3 install numpy>=1.21.0
pip3 install PyYAML>=5.4.0
pip3 install matplotlib>=3.4.0
pip3 install seaborn>=0.11.0
pip3 install tqdm>=4.62.0
pip3 install scipy>=1.7.0
pip3 install shap>=0.41.0

# Install Flwr and related packages
echo "Installing Flwr federated learning framework..."
pip3 install flwr>=1.6.0
pip3 install flwr-datasets>=0.0.2

# Install IoT and resource monitoring dependencies
echo "Installing IoT support packages..."
pip3 install psutil>=5.9.0
pip3 install "ray[default]">=2.8.0

# Install additional dependencies
pip3 install tensorboard>=2.10.0
pip3 install protobuf>=3.19.0

echo "Installation completed!"

# Verify installation
echo "Verifying installation..."
python3 -c "import flwr; print(f'Flwr version: {flwr.__version__}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')"

echo "Setup complete! You can now run the simulation with:"
echo "cd TRUST_MCNet_Codebase"
echo "python3 scripts/flwr_simulation.py --config config/config.yaml"

# Check for GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "False"; then
    echo "Note: CUDA not available. Training will run on CPU (suitable for IoT simulation)."
fi

echo "IoT Federated Learning setup with Flwr is ready!"
