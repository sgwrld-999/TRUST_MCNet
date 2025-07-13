# TRUST-MCNet Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sgwrld-999/TRUST_MCNet.git
   cd TRUST_MCNet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch, numpy, pandas; print('Dependencies installed successfully!')"
   ```

## Running Your First Simulation

### Basic Usage
```bash
# Run with default settings (5 clients, 5 rounds)
python main.py

# Or run directly from examples
python examples/start_simulation.py
```

### Custom Configuration
```bash
# Run with custom parameters
python examples/start_simulation.py --clients 10 --rounds 10 --verbose

# Use custom trust threshold
python examples/start_simulation.py --trust-threshold 0.8
```

## Project Structure Overview

- **`src/trust_mcnet/`** - Main framework code
- **`examples/`** - Ready-to-run examples and demos
- **`config/`** - Configuration files for different scenarios
- **`data/`** - IoT datasets and data processing
- **`docs/`** - Documentation, reports, and diagrams
- **`tests/`** - Test suite for validation

## Key Features

✅ **Federated Learning** - Distributed training across IoT clients  
✅ **Trust Mechanisms** - Advanced client reliability evaluation  
✅ **IoT Optimization** - Resource-aware training for edge devices  
✅ **Multi-Dataset Support** - CIC-IoMT, Edge-IIoT, MedBIoT, and more  
✅ **Comprehensive Logging** - Detailed experiment tracking  
✅ **Modern Architecture** - Clean, maintainable codebase  

## Next Steps

1. Check out `examples/start_simulation.py` for the main simulation
2. Review `config/` for different configuration options
3. Read `docs/reports/` for detailed analysis and reports
4. Explore `src/trust_mcnet/` to understand the framework architecture

## Support

For issues and questions:
- Check the documentation in `docs/`
- Review the comprehensive implementation report
- Examine the code examples in `examples/`

Happy experimenting with TRUST-MCNet! 🚀
