"""
OmegaConf structured configuration schemas for TRUST-MCNet.

This module defines dataclass-based schemas for automatic validation
of configuration parameters at startup.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Dataset configuration schema."""
    name: str = MISSING
    path: str = MISSING
    num_clients: int = MISSING
    eval_fraction: float = 0.2
    batch_size: int = 32
    val_ratio: float = 0.1
    
    # Transform settings
    transforms: Dict[str, Any] = field(default_factory=dict)
    
    # Partitioning settings
    partitioning: str = "iid"
    dirichlet_alpha: float = 0.5
    
    # Binary classification settings
    binary_classification: Optional[Dict[str, Any]] = None
    
    # Validation constraints
    min_samples_per_client: int = 10
    max_samples_per_client: int = 10000


@dataclass
class ModelConfig:
    """Model configuration schema."""
    name: str = MISSING
    input_dim: int = MISSING
    output_dim: int = MISSING
    hidden_dims: Optional[List[int]] = None
    
    # LSTM specific
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    
    # Model hyperparameters
    dropout: float = 0.0
    batch_norm: bool = False


@dataclass
class TrainingConfig:
    """Training configuration schema."""
    epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    
    # Optimizer specific parameters
    momentum: float = 0.9  # For SGD
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])  # For Adam
    
    # Learning rate scheduling
    lr_scheduler: Optional[str] = None
    lr_decay: float = 0.1
    lr_patience: int = 10


@dataclass
class RayConfig:
    """Ray configuration schema."""
    num_cpus: int = 4
    num_gpus: int = 0
    object_store_memory: int = 1000000000
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8265
    ignore_reinit_error: bool = True


@dataclass
class SimulationConfig:
    """Simulation environment configuration schema."""
    client_resources: Dict[str, float] = field(default_factory=lambda: {"num_cpus": 1, "num_gpus": 0})


@dataclass
class EnvConfig:
    """Environment configuration schema."""
    device: str = "auto"
    ray: RayConfig = field(default_factory=RayConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


@dataclass
class FederatedConfig:
    """Federated learning configuration schema."""
    num_rounds: int = 3
    fraction_fit: float = 0.8
    fraction_evaluate: float = 0.2
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2


@dataclass
class StrategyConfig:
    """Federated learning strategy configuration schema."""
    name: str = "fedavg"
    
    # FedProx specific
    proximal_mu: float = 1.0
    
    # FedAdam specific
    eta: float = 1e-3
    eta_l: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau: float = 1e-9


@dataclass
class TrustConfig:
    """Trust evaluation configuration schema."""
    mode: str = "hybrid"
    threshold: float = 0.5
    
    # Weight configurations for hybrid mode
    weights: Dict[str, float] = field(default_factory=lambda: {
        "cosine": 0.4,
        "entropy": 0.3,
        "reputation": 0.3
    })
    
    # Trust evaluation parameters
    entropy_bins: int = 50
    reputation_decay: float = 0.9
    min_trust_score: float = 0.1


@dataclass
class LoggingConfig:
    """Logging configuration schema."""
    level: str = "INFO"
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_to_file: bool = False
    log_file: Optional[str] = None


@dataclass
class HydraConfig:
    """Hydra configuration schema."""
    run: Dict[str, str] = field(default_factory=lambda: {
        "dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    })
    sweep: Dict[str, str] = field(default_factory=lambda: {
        "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "subdir": "${hydra:job.num}"
    })
    job: Dict[str, bool] = field(default_factory=lambda: {"chdir": False})


@dataclass
class Config:
    """Root configuration schema."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)
