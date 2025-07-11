# trust_module/rho_adaptor.py

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple

class RhoWeightAdaptor:
    def __init__(self, eta: float = 0.1):
        self.previous_theta = np.array([1/3, 1/3, 1/3])  # Initial weights
        self.eta = eta

    def softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))

    def update_weights(self, cosine_scores: np.ndarray, 
                       entropy_scores: np.ndarray, 
                       reputation_scores: np.ndarray, 
                       delta_acc: float) -> Tuple[float, float, float]:
        # Compute Spearman correlation
        rho, _ = spearmanr([cosine_scores, entropy_scores, reputation_scores], axis=1)
        
        # Update theta
        self.previous_theta = self.softplus(self.previous_theta + self.eta * rho)
        self.previous_theta /= self.previous_theta.sum()  # Normalize to sum to 1
        
        return tuple(self.previous_theta)