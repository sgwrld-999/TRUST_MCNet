# trust_module/rho_adaptor.py

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple

class RhoWeightAdaptor:
    def __init__(self, eta: float):
        self.previous_theta = np.array([1.0, 0.0, 0.0])  # Initial weights
        self.eta = eta

    def softplus(self, x):
        return np.log1p(np.exp(x))

    def update_weights(self, cosine_scores: np.ndarray, entropy_scores: np.ndarray, reputation_scores: np.ndarray, delta_acc: float) -> Tuple[float, float, float]:
        # Compute Spearman correlation
        rho, _ = spearmanr([cosine_scores, entropy_scores, reputation_scores], axis=1)
        
        # Update theta
        self.previous_theta += self.eta * rho
        theta = self.softplus(self.previous_theta)
        theta /= theta.sum()  # Normalize to ensure α + β + γ = 1
        
        return tuple(theta)