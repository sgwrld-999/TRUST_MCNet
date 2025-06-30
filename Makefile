# Makefile for TRUST MCNet

.PHONY: help install install-dev clean test lint format run example

# Default target
help:
	@echo "TRUST MCNet - Federated Learning Framework"
	@echo "Available commands:"
	@echo "  install         - Install dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  install-flwr    - Install Flwr federated learning framework"
	@echo "  clean           - Clean up generated files"
	@echo "  test            - Run tests"
	@echo "  test-flwr       - Test Flwr installation"
	@echo "  lint            - Run linting"
	@echo "  format          - Format code"
	@echo "  run             - Run the main application"
	@echo "  run-baseline    - Run baseline experiment with random weights"
	@echo "  run-baseline-custom - Run baseline with custom parameters"
	@echo "  run-mnist-baseline  - Run MNIST verification baseline"
	@echo "  run-flwr        - Run Flwr simulation (default params)"
	@echo "  run-flwr-custom - Run Flwr simulation (custom params)"
	@echo "  run-flwr-noniid - Run Flwr simulation (non-IID data)"
	@echo "  example         - Run basic example"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest black flake8 mypy

# Install Flwr federated learning dependencies
install-flwr:
	chmod +x setup_flwr.sh
	./setup_flwr.sh

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 TRUST_MCNet_Codebase/
	mypy TRUST_MCNet_Codebase/

# Format code
format:
	black TRUST_MCNet_Codebase/
	black examples/

# Run the main application
run:
	python TRUST_MCNet_Codebase/main.py

# Run basic example
example:
	python examples/basic_example.py

# Install package in development mode
dev-install:
	pip install -e .

# Build package
build:
	python setup.py sdist bdist_wheel

# Upload to PyPI (use with caution)
upload-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	twine upload dist/*

# Run Flwr simulation with default parameters
run-flwr:
	cd TRUST_MCNet_Codebase && python scripts/flwr_simulation.py --config config/config.yaml

# Run Flwr simulation with custom parameters
run-flwr-custom:
	cd TRUST_MCNet_Codebase && python scripts/flwr_simulation.py --config config/config.yaml --clients 10 --rounds 50

# Run IoT simulation with non-IID data
run-flwr-noniid:
	cd TRUST_MCNet_Codebase && python scripts/flwr_simulation.py --config config/config.yaml --data-distribution non_iid

# Test Flwr installation
test-flwr:
	python3 -c "import flwr; print(f'Flwr version: {flwr.__version__}')"
	python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
	python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')"

# Run baseline experiment with random weights
run-baseline:
	cd TRUST_MCNet_Codebase && python scripts/baseline_experiment.py --config config/config.yaml

# Run baseline with custom parameters
run-baseline-custom:
	cd TRUST_MCNet_Codebase && python scripts/baseline_experiment.py --config config/config.yaml --rounds 30 --clients 5

# Run baseline with MNIST verification
run-mnist-baseline:
	cd TRUST_MCNet_Codebase && python scripts/baseline_experiment.py --config config/config.yaml --experiment-name mnist_verification
