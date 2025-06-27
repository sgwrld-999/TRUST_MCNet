# Makefile for TRUST MCNet

.PHONY: help install install-dev clean test lint format run example

# Default target
help:
	@echo "TRUST MCNet - Federated Learning Framework"
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  clean       - Clean up generated files"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  run         - Run the main application"
	@echo "  example     - Run basic example"

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest black flake8 mypy

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
