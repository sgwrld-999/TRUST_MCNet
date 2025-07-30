#STRATEGY CELL - 2
import warnings
import numpy as np
from math import sqrt
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional, Callable

from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitIns,
    EvaluateIns,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

warnings.filterwarnings("ignore")


class TrustMCStrategy(FedAvg):
    def __init__(
        self,
        *,
        # Hy-perparameters for the Bayesian-mirror update
        percentile: float = 40,
        eta0: float = 0.10,
        lam: float = 0.2,
        kappa: float = 5.0,
        # Hy-perparameters for softmax weighting
        temp0: float = 1.0,
        temp_min: float = 0.3,
        temp_decay: float = 0.1,
        # Hy-perparameters for adaptive client LR
        min_lr: float = 1e-3,
        max_lr: float = 1e-1,
        # All other FedAvg parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, List[np.ndarray], Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        # These two are *not* used because we override configure_fit/evaluate
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=None,
            on_evaluate_config_fn=None,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # State for Bayesian-mirror update
        self.theta: np.ndarray = np.array([1/3, 1/3, 1/3])
        self.t: int = 0
        self.lam = lam
        self.eta0 = eta0
        self.kappa = kappa

        # State for dynamic weighting
        self.percentile = percentile
        self.temp0 = temp0
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        # State for adaptive per-client LR
        self.min_lr = min_lr
        self.max_lr = max_lr

        # Track previous global accuracy to compute ΔAcc
        self.prev_global_accuracy: float = 0.0

        # Will hold last-round trust scores {client_id: T_i}
        self.trust_scores: Dict[str, float] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # If no results, fallback to FedAvg
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        # 1) Extract per-client metrics & ΔAcc
        metric_names = ["cos", "ent", "rep"]
        M_list: List[List[float]] = []
        delta_acc_list: List[float] = []

        for client, fit_res in results:
            # collect [cos, ent, rep]
            M_list.append([fit_res.metrics.get(name, 0.0) for name in metric_names])
            # ΔAcc = current client accuracy − last global accuracy
            acc = float(fit_res.metrics.get("accuracy", 0.0))
            delta_acc_list.append(acc - self.prev_global_accuracy)

        M = np.array(M_list)                 # shape (n_clients, 3)
        delta_acc = np.array(delta_acc_list) # shape (n_clients,)

        # 2) Compute Spearman ρ for each metric dimension
        rhos: List[float] = []
        for j in range(M.shape[1]):
            ρ, _ = spearmanr(M[:, j], delta_acc)
            rhos.append(0.0 if np.isnan(ρ) else float(ρ))
        ρ = np.array(rhos)                   # shape (3,)

        # 3) Bayesian-mirror update of θ
        self.t += 1
        s = (ρ + 1.0) / 2.0
        θ_bar = (1 - self.lam) * self.theta + self.lam * s
        η_t = self.eta0 / sqrt(self.t)
        g = np.exp(η_t * ρ)
        θ_new = θ_bar * g
        θ_new /= θ_new.sum()
        self.theta = θ_new

        # 4) Compute trust scores T_i = θ · metrics_i
        T = M.dot(self.theta)  # shape (n_clients,)

        # 5) Store trust_scores for use in configure_* and on_fit_config_fn
        for (client, _), score in zip(results, T):
            self.trust_scores[client.cid] = float(score)

        # 6) Determine dynamic threshold & trusted mask
        τ = float(np.percentile(T, self.percentile))
        trusted_mask = T >= τ

        # 7) Softmax weighting among trusted clients
        temp_t = max(self.temp0 - self.temp_decay * (server_round - 1), self.temp_min)
        w = np.zeros_like(T)
        if trusted_mask.any():
            exp_scores = np.exp(T[trusted_mask] / temp_t)
            w[trusted_mask] = exp_scores / exp_scores.sum()

        # 8) Weighted aggregation
        client_params = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        # Initialize new_weights = zeros_like first client
        new_weights = [np.zeros_like(arr) for arr in client_params[0]]
        for weight_i, arr in zip(w, client_params):
            for idx in range(len(arr)):
                new_weights[idx] += weight_i * arr[idx]

        # 9) Update prev_global_accuracy
        accs = np.array([float(fit_res.metrics.get("accuracy", 0.0)) for _, fit_res in results])
        self.prev_global_accuracy = float((accs * w).sum())

        # 10) Return aggregated parameters
        return ndarrays_to_parameters(new_weights), {}

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Fallback if no trust scores yet
        if not self.trust_scores:
            return super().configure_fit(server_round, parameters, client_manager)

        # Sample clients as usual
        sample_size, min_clients = self.num_fit_clients(client_manager.num_available())
        sampled = client_manager.sample(num_clients=sample_size, min_num_clients=min_clients)

        # Determine threshold
        scores = np.array(list(self.trust_scores.values()), dtype=float)
        τ = float(np.percentile(scores, self.percentile))

        # Filter trusted
        trusted = [c for c in sampled if self.trust_scores.get(c.cid, 0.0) >= τ]
        if not trusted:
            # fallback to FedAvg sampling
            return super().configure_fit(server_round, parameters, client_manager)

        # Build per-client fit configs
        cfg_map = self._on_fit_config_fn(server_round)
        fit_ins: List[Tuple[ClientProxy, FitIns]] = []
        for client in trusted:
            cfg = cfg_map.get(client.cid, {})
            fit_ins.append((client, FitIns(parameters, cfg)))
        return fit_ins

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Fallback if no trust scores yet
        if not self.trust_scores:
            return super().configure_evaluate(server_round, parameters, client_manager)

        # Sample clients as usual
        sample_size, min_clients = self.num_evaluation_clients(client_manager.num_available())
        sampled = client_manager.sample(num_clients=sample_size, min_num_clients=min_clients)

        # Determine threshold
        scores = np.array(list(self.trust_scores.values()), dtype=float)
        τ = float(np.percentile(scores, self.percentile))

        # Filter trusted
        trusted = [c for c in sampled if self.trust_scores.get(c.cid, 0.0) >= τ]
        if not trusted:
            return super().configure_evaluate(server_round, parameters, client_manager)

        # Build EvaluateIns (no per-client config by default)
        eval_ins: List[Tuple[ClientProxy, EvaluateIns]] = []
        default_cfg = {}
        if self.on_evaluate_config_fn is not None:
            default_cfg = self.on_evaluate_config_fn(server_round)
        for client in trusted:
            eval_ins.append((client, EvaluateIns(parameters, default_cfg)))
        return eval_ins

    def _on_fit_config_fn(self, server_round: int) -> Dict[str, Dict[str, float]]:
        """Produce per-client fit config mapping client_id → {'learning_rate': lr}."""
        if not self.trust_scores:
            return {}

        scores = np.array(list(self.trust_scores.values()), dtype=float)
        s_min, s_max = float(scores.min()), float(scores.max())
        span = max(s_max - s_min, 1e-8)

        cfg_map: Dict[str, Dict[str, float]] = {}
        for cid, score in self.trust_scores.items():
            rel = (score - s_min) / span
            lr = self.min_lr + rel * (self.max_lr - self.min_lr)
            cfg_map[cid] = {"learning_rate": lr}
        return cfg_map
