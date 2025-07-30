# SETUP CELL
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import sqrt
from scipy.stats import spearmanr

# Import Flower federated learning framework
import flwr as fl
from flwr.common import Parameters
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("Setup complete. PyTorch version:", torch.__version__, "Flower version:", fl.__version__)