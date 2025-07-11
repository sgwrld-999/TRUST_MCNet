# trust_module/rho_adaptor.py

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple

class RhoWeightAdaptor:
    def __init__(self, eta: float):
        self.previous_theta = np.array([1.0, 0.0, 0.0])  # Initial weights for α, β, γ
        self.eta = eta

    def update_weights(self, cosine_scores: np.ndarray, entropy_scores: np.ndarray, 
                       reputation_scores: np.ndarray, delta_acc: float) -> Tuple[float, float, float]:
        # Compute Spearman correlation
        rho, _ = spearmanr([cosine_scores, entropy_scores, reputation_scores], axis=1)
        
        # Update theta using softplus
        self.previous_theta += self.eta * rho
        theta = np.maximum(0, self.previous_theta)  # Softplus activation
        theta /= theta.sum()  # Normalize to sum to 1
        
        return tuple(theta)  # Return α, β, γ