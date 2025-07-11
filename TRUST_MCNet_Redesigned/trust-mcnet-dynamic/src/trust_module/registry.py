# trust_module/rho_adaptor.py

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple

class RhoWeightAdaptor:
    def __init__(self, initial_theta: np.ndarray, learning_rate: float):
        self.previous_theta = initial_theta
        self.η = learning_rate

    def update_weights(self, cosine_scores: np.ndarray, 
                       entropy_scores: np.ndarray, 
                       reputation_scores: np.ndarray, 
                       delta_acc: float) -> Tuple[float, float, float]:
        # Compute Spearman correlation
        rho, _ = spearmanr([cosine_scores, entropy_scores, reputation_scores], axis=1)
        
        # Update theta using softplus
        theta = np.maximum(0, self.previous_theta + self.η * rho)
        theta /= theta.sum()  # Normalize to ensure α + β + γ = 1
        
        self.previous_theta = theta  # Update previous_theta for next round
        return tuple(theta)