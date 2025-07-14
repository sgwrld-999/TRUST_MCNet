"""
Enhanced Configuration Management for TRUST-MCNet

This module provides improved configuration schemas with comprehensive validation,
type safety, and better defaults for the federated learning framework.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
from omegaconf import MISSING
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Supported dataset types."""
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    EDGE_IIOT = "edge_iiot"
    TON_IOT = "ton_iot"
    MEDBIOT = "medbiot"
    CUSTOM_CSV = "custom_csv"


class PartitioningStrategy(Enum):
    """Data partitioning strategies for federated learning."""
    IID = "iid"
    DIRICHLET = "dirichlet"
    PATHOLOGICAL = "pathological"
    CLUSTERED = "clustered"


class ModelType(Enum):
    """Supported model architectures."""
    MLP = "mlp"
    LSTM = "lstm"
    CNN = "cnn"


class TrustMode(Enum):
    """Trust evaluation modes."""
    COSINE = "cosine"
    ENTROPY = "entropy"
    REPUTATION = "reputation"
    HYBRID = "hybrid"


class OptimizerType(Enum):
    """Supported optimizers."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class DatasetConfig:
    """
    Comprehensive dataset configuration with validation.
    
    This configuration supports multiple dataset types and federated learning
    partitioning strategies with proper validation and sensible defaults.
    """
    
    # Required core configuration
    name: DatasetType = MISSING
    path: Union[str, Path] = MISSING
    num_clients: int = MISSING
    
    # Data processing configuration  
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training and evaluation"}
    )
    eval_fraction: float = field(
        default=0.2,
        metadata={"help": "Fraction of data reserved for final evaluation"}
    )
    val_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio for each client"}
    )
    
    # Partitioning configuration
    partitioning: PartitioningStrategy = field(
        default=PartitioningStrategy.IID,
        metadata={"help": "Data partitioning strategy across clients"}
    )
    
    # Strategy-specific parameters
    dirichlet_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha parameter for Dirichlet partitioning (lower = more non-IID)"}
    )
    pathological_shards: int = field(
        default=2,
        metadata={"help": "Number of shards per client for pathological partitioning"}
    )
    cluster_overlap: float = field(
        default=0.1,
        metadata={"help": "Overlap ratio for clustered partitioning"}
    )
    
    # Data preprocessing
    transforms: Dict[str, Any] = field(
        default_factory=lambda: {
            "normalize": True,
            "standardize": False,
            "augment": False
        },
        metadata={"help": "Data preprocessing transformations"}
    )
    
    # Quality constraints
    min_samples_per_client: int = field(
        default=10,
        metadata={"help": "Minimum samples required per client"}
    )
    max_samples_per_client: int = field(
        default=10000,
        metadata={"help": "Maximum samples allowed per client"}
    )
    
    # Advanced options
    stratified_split: bool = field(
        default=True,
        metadata={"help": "Use stratified splitting to preserve class distributions"}
    )
    random_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for reproducible partitioning"}
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_required_fields()
        self._validate_numerical_constraints()
        self._validate_partitioning_config()
        self._convert_types()
        
        logger.info(f"Initialized DatasetConfig: {self.get_summary()}")
    
    def _validate_required_fields(self) -> None:
        """Validate that all required fields are provided."""
        if self.name == MISSING:
            raise ConfigurationError("Dataset name is required")
        if self.path == MISSING:
            raise ConfigurationError("Dataset path is required")
        if self.num_clients == MISSING:
            raise ConfigurationError("Number of clients is required")
    
    def _validate_numerical_constraints(self) -> None:
        """Validate numerical field constraints."""
        constraints = [
            (0 < self.eval_fraction < 1, f"eval_fraction must be in (0,1), got {self.eval_fraction}"),
            (0 < self.val_ratio < 1, f"val_ratio must be in (0,1), got {self.val_ratio}"),
            (self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"),
            (self.num_clients > 0, f"num_clients must be positive, got {self.num_clients}"),
            (self.min_samples_per_client > 0, f"min_samples_per_client must be positive"),
            (self.max_samples_per_client >= self.min_samples_per_client, 
             f"max_samples_per_client must be >= min_samples_per_client"),
            (self.dirichlet_alpha > 0, f"dirichlet_alpha must be positive, got {self.dirichlet_alpha}"),
            (self.pathological_shards > 0, f"pathological_shards must be positive"),
            (0 <= self.cluster_overlap <= 1, f"cluster_overlap must be in [0,1], got {self.cluster_overlap}")
        ]
        
        for condition, error_msg in constraints:
            if not condition:
                raise ConfigurationError(error_msg)
    
    def _validate_partitioning_config(self) -> None:
        """Validate partitioning-specific configuration."""
        if self.partitioning == PartitioningStrategy.PATHOLOGICAL:
            if self.pathological_shards >= 10:  # Reasonable upper bound
                logger.warning(f"Large number of pathological shards: {self.pathological_shards}")
        
        if self.partitioning == PartitioningStrategy.DIRICHLET:
            if self.dirichlet_alpha > 10:
                logger.warning(f"Large Dirichlet alpha may result in near-IID distribution: {self.dirichlet_alpha}")
    
    def _convert_types(self) -> None:
        """Convert and normalize types."""
        # Convert string enums if necessary
        if isinstance(self.name, str):
            self.name = DatasetType(self.name)
        if isinstance(self.partitioning, str):
            self.partitioning = PartitioningStrategy(self.partitioning)
        
        # Convert path to Path object
        if isinstance(self.path, str):
            self.path = Path(self.path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            "dataset": self.name.value,
            "num_clients": self.num_clients,
            "partitioning": self.partitioning.value,
            "batch_size": self.batch_size,
            "eval_fraction": self.eval_fraction
        }
    
    def is_non_iid(self) -> bool:
        """Check if configuration results in non-IID data distribution."""
        return self.partitioning != PartitioningStrategy.IID


@dataclass
class ModelConfig:
    """Enhanced model configuration with comprehensive options."""
    
    # Core model configuration
    architecture: ModelType = MISSING
    input_dim: int = MISSING
    output_dim: int = MISSING
    
    # Architecture-specific parameters
    hidden_dims: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Hidden layer dimensions for MLP"}
    )
    
    # LSTM-specific parameters
    lstm_hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden dimension for LSTM layers"}
    )
    lstm_num_layers: int = field(
        default=2,
        metadata={"help": "Number of LSTM layers"}
    )
    lstm_bidirectional: bool = field(
        default=False,
        metadata={"help": "Use bidirectional LSTM"}
    )
    
    # CNN-specific parameters (for future extension)
    cnn_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128],
        metadata={"help": "CNN channel dimensions"}
    )
    cnn_kernel_sizes: List[int] = field(
        default_factory=lambda: [3, 3, 3],
        metadata={"help": "CNN kernel sizes"}
    )
    
    # Regularization
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for regularization"}
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Use batch normalization instead of layer normalization"}
    )
    
    # Activation and initialization
    activation: Literal["relu", "leaky_relu", "gelu", "elu"] = field(
        default="relu",
        metadata={"help": "Activation function"}
    )
    weight_init: Literal["xavier", "kaiming", "normal"] = field(
        default="xavier",
        metadata={"help": "Weight initialization scheme"}
    )
    
    def __post_init__(self):
        """Validate model configuration."""
        self._validate_required_fields()
        self._validate_parameters()
        self._set_defaults()
    
    def _validate_required_fields(self) -> None:
        """Validate required fields."""
        if self.architecture == MISSING:
            raise ConfigurationError("Model architecture is required")
        if self.input_dim == MISSING:
            raise ConfigurationError("Input dimension is required")
        if self.output_dim == MISSING:
            raise ConfigurationError("Output dimension is required")
    
    def _validate_parameters(self) -> None:
        """Validate parameter constraints."""
        if self.input_dim <= 0:
            raise ConfigurationError(f"Input dimension must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ConfigurationError(f"Output dimension must be positive, got {self.output_dim}")
        if not (0 <= self.dropout_rate < 1):
            raise ConfigurationError(f"Dropout rate must be in [0,1), got {self.dropout_rate}")
        if self.lstm_hidden_dim <= 0:
            raise ConfigurationError("LSTM hidden dimension must be positive")
        if self.lstm_num_layers <= 0:
            raise ConfigurationError("Number of LSTM layers must be positive")
    
    def _set_defaults(self) -> None:
        """Set architecture-specific defaults."""
        if isinstance(self.architecture, str):
            self.architecture = ModelType(self.architecture)
        
        if self.architecture == ModelType.MLP and self.hidden_dims is None:
            # Default MLP architecture for IoT anomaly detection
            self.hidden_dims = [1024, 512, 256, 128, 64, 32, 16]


@dataclass
class TrainingConfig:
    """Enhanced training configuration with comprehensive options."""
    
    # Core training parameters
    local_epochs: int = field(
        default=1,
        metadata={"help": "Number of local training epochs per round"}
    )
    learning_rate: float = field(
        default=0.001,
        metadata={"help": "Learning rate for local training"}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "L2 regularization weight decay"}
    )
    
    # Optimizer configuration
    optimizer: OptimizerType = field(
        default=OptimizerType.ADAM,
        metadata={"help": "Optimizer for local training"}
    )
    
    # Optimizer-specific parameters
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for SGD optimizer"}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 parameter for Adam optimizer"}
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 parameter for Adam optimizer"}
    )
    
    # Learning rate scheduling
    use_lr_scheduler: bool = field(
        default=False,
        metadata={"help": "Enable learning rate scheduling"}
    )
    lr_scheduler_type: Literal["step", "exponential", "cosine"] = field(
        default="step",
        metadata={"help": "Learning rate scheduler type"}
    )
    lr_decay_factor: float = field(
        default=0.1,
        metadata={"help": "Learning rate decay factor"}
    )
    lr_decay_patience: int = field(
        default=10,
        metadata={"help": "Patience for learning rate decay"}
    )
    
    # Training enhancements
    gradient_clipping: Optional[float] = field(
        default=None,
        metadata={"help": "Gradient clipping threshold (None = disabled)"}
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Enable early stopping based on validation loss"}
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Patience for early stopping"}
    )
    
    # Resource management
    enable_gpu_optimization: bool = field(
        default=True,
        metadata={"help": "Enable GPU optimizations if available"}
    )
    enable_memory_cleanup: bool = field(
        default=True,
        metadata={"help": "Enable automatic memory cleanup"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum retries for failed training"}
    )
    retry_delay: float = field(
        default=1.0,
        metadata={"help": "Delay between retries (seconds)"}
    )
    
    def __post_init__(self):
        """Validate training configuration."""
        self._validate_parameters()
        self._convert_types()
    
    def _validate_parameters(self) -> None:
        """Validate training parameters."""
        constraints = [
            (self.local_epochs > 0, "Local epochs must be positive"),
            (self.learning_rate > 0, "Learning rate must be positive"),
            (self.weight_decay >= 0, "Weight decay must be non-negative"),
            (0 < self.momentum < 1, "Momentum must be in (0,1)"),
            (0 < self.beta1 < 1, "Beta1 must be in (0,1)"),
            (0 < self.beta2 < 1, "Beta2 must be in (0,1)"),
            (0 < self.lr_decay_factor < 1, "LR decay factor must be in (0,1)"),
            (self.lr_decay_patience > 0, "LR decay patience must be positive"),
            (self.early_stopping_patience > 0, "Early stopping patience must be positive"),
            (self.max_retries >= 0, "Max retries must be non-negative"),
            (self.retry_delay >= 0, "Retry delay must be non-negative")
        ]
        
        for condition, error_msg in constraints:
            if not condition:
                raise ConfigurationError(error_msg)
        
        if self.gradient_clipping is not None and self.gradient_clipping <= 0:
            raise ConfigurationError("Gradient clipping threshold must be positive")
    
    def _convert_types(self) -> None:
        """Convert string types to enums."""
        if isinstance(self.optimizer, str):
            self.optimizer = OptimizerType(self.optimizer)


@dataclass
class TrustConfig:
    """Enhanced trust evaluation configuration."""
    
    # Core trust configuration
    trust_mode: TrustMode = field(
        default=TrustMode.HYBRID,
        metadata={"help": "Trust evaluation mode"}
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "Trust threshold for client selection"}
    )
    
    # Dynamic adaptation
    use_dynamic_weights: bool = field(
        default=True,
        metadata={"help": "Enable dynamic weight adaptation"}
    )
    learning_rate: float = field(
        default=0.01,
        metadata={"help": "Learning rate for weight adaptation"}
    )
    
    # Component weights (for static mode)
    cosine_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for cosine similarity component"}
    )
    entropy_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for entropy component"}
    )
    reputation_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for reputation component"}
    )
    
    # Trust calculation parameters
    similarity_threshold: float = field(
        default=0.1,
        metadata={"help": "Minimum similarity threshold"}
    )
    max_entropy_estimate: float = field(
        default=2.3,
        metadata={"help": "Maximum entropy estimate for normalization"}
    )
    reputation_window_size: int = field(
        default=10,
        metadata={"help": "Window size for reputation calculation"}
    )
    reputation_decay_factor: float = field(
        default=0.9,
        metadata={"help": "Decay factor for historical reputation"}
    )
    
    def __post_init__(self):
        """Validate trust configuration."""
        self._validate_parameters()
        self._convert_types()
        self._normalize_weights()
    
    def _validate_parameters(self) -> None:
        """Validate trust parameters."""
        constraints = [
            (0 <= self.threshold <= 1, "Trust threshold must be in [0,1]"),
            (self.learning_rate > 0, "Learning rate must be positive"),
            (self.cosine_weight >= 0, "Cosine weight must be non-negative"),
            (self.entropy_weight >= 0, "Entropy weight must be non-negative"),
            (self.reputation_weight >= 0, "Reputation weight must be non-negative"),
            (self.similarity_threshold >= 0, "Similarity threshold must be non-negative"),
            (self.max_entropy_estimate > 0, "Max entropy estimate must be positive"),
            (self.reputation_window_size > 0, "Reputation window size must be positive"),
            (0 < self.reputation_decay_factor < 1, "Reputation decay factor must be in (0,1)")
        ]
        
        for condition, error_msg in constraints:
            if not condition:
                raise ConfigurationError(error_msg)
    
    def _convert_types(self) -> None:
        """Convert string types to enums."""
        if isinstance(self.trust_mode, str):
            self.trust_mode = TrustMode(self.trust_mode)
    
    def _normalize_weights(self) -> None:
        """Normalize component weights to sum to 1."""
        total_weight = self.cosine_weight + self.entropy_weight + self.reputation_weight
        if total_weight > 0:
            self.cosine_weight /= total_weight
            self.entropy_weight /= total_weight
            self.reputation_weight /= total_weight
        else:
            # Default equal weights
            self.cosine_weight = self.entropy_weight = self.reputation_weight = 1/3


@dataclass
class FederatedConfig:
    """Enhanced federated learning configuration."""
    
    # Core FL parameters
    num_rounds: int = field(
        default=5,
        metadata={"help": "Number of federated learning rounds"}
    )
    fraction_fit: float = field(
        default=0.8,
        metadata={"help": "Fraction of clients selected for training"}
    )
    fraction_evaluate: float = field(
        default=0.2,
        metadata={"help": "Fraction of clients selected for evaluation"}
    )
    
    # Client selection constraints
    min_fit_clients: int = field(
        default=2,
        metadata={"help": "Minimum number of clients for training"}
    )
    min_evaluate_clients: int = field(
        default=1,
        metadata={"help": "Minimum number of clients for evaluation"}
    )
    min_available_clients: int = field(
        default=2,
        metadata={"help": "Minimum number of available clients"}
    )
    
    # Aggregation strategy
    strategy: Literal["fedavg", "fedprox", "fedadam", "fednova"] = field(
        default="fedavg",
        metadata={"help": "Federated learning strategy"}
    )
    
    # Strategy-specific parameters
    fedprox_mu: float = field(
        default=0.01,
        metadata={"help": "Proximal term coefficient for FedProx"}
    )
    fedadam_eta: float = field(
        default=1e-3,
        metadata={"help": "Server learning rate for FedAdam"}
    )
    fedadam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 parameter for FedAdam"}
    )
    fedadam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 parameter for FedAdam"}
    )
    
    # Server configuration
    accept_failures: bool = field(
        default=True,
        metadata={"help": "Accept client failures during training"}
    )
    server_address: str = field(
        default="0.0.0.0:8080",
        metadata={"help": "Server address for Flower server"}
    )
    
    def __post_init__(self):
        """Validate federated learning configuration."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate federated learning parameters."""
        constraints = [
            (self.num_rounds > 0, "Number of rounds must be positive"),
            (0 < self.fraction_fit <= 1, "Fraction fit must be in (0,1]"),
            (0 < self.fraction_evaluate <= 1, "Fraction evaluate must be in (0,1]"),
            (self.min_fit_clients > 0, "Min fit clients must be positive"),
            (self.min_evaluate_clients > 0, "Min evaluate clients must be positive"),
            (self.min_available_clients > 0, "Min available clients must be positive"),
            (self.min_fit_clients <= self.min_available_clients, 
             "Min fit clients must be <= min available clients"),
            (self.min_evaluate_clients <= self.min_available_clients,
             "Min evaluate clients must be <= min available clients"),
            (self.fedprox_mu >= 0, "FedProx mu must be non-negative"),
            (self.fedadam_eta > 0, "FedAdam eta must be positive"),
            (0 < self.fedadam_beta1 < 1, "FedAdam beta1 must be in (0,1)"),
            (0 < self.fedadam_beta2 < 1, "FedAdam beta2 must be in (0,1)")
        ]
        
        for condition, error_msg in constraints:
            if not condition:
                raise ConfigurationError(error_msg)


@dataclass
class ExperimentConfig:
    """Root configuration combining all components."""
    
    # Component configurations
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    trust: TrustConfig = MISSING
    federated: FederatedConfig = MISSING
    
    # Experiment metadata
    experiment_name: str = field(
        default="trust_mcnet_experiment",
        metadata={"help": "Name of the experiment"}
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Description of the experiment"}
    )
    tags: List[str] = field(
        default_factory=list,
        metadata={"help": "Tags for experiment organization"}
    )
    
    # Output configuration
    output_dir: Path = field(
        default=Path("./outputs"),
        metadata={"help": "Output directory for results"}
    )
    save_models: bool = field(
        default=True,
        metadata={"help": "Save trained models"}
    )
    save_metrics: bool = field(
        default=True,
        metadata={"help": "Save detailed metrics"}
    )
    
    # Logging and monitoring
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO",
        metadata={"help": "Logging level"}
    )
    enable_tensorboard: bool = field(
        default=False,
        metadata={"help": "Enable TensorBoard logging"}
    )
    enable_mlflow: bool = field(
        default=False,
        metadata={"help": "Enable MLflow tracking"}
    )
    
    def __post_init__(self):
        """Validate and setup experiment configuration."""
        self._validate_configuration()
        self._setup_output_directory()
    
    def _validate_configuration(self) -> None:
        """Validate the complete experiment configuration."""
        # Check required components
        required_components = ['dataset', 'model', 'training', 'trust', 'federated']
        missing_components = []
        
        for component in required_components:
            if getattr(self, component) == MISSING:
                missing_components.append(component)
        
        if missing_components:
            raise ConfigurationError(f"Missing required configuration components: {missing_components}")
        
        # Cross-component validation
        self._validate_cross_component_consistency()
    
    def _validate_cross_component_consistency(self) -> None:
        """Validate consistency across different configuration components."""
        # Ensure model input/output dimensions match dataset
        if hasattr(self.model, 'input_dim') and hasattr(self.dataset, 'input_dim'):
            if self.model.input_dim != self.dataset.input_dim:
                logger.warning(f"Model input dim ({self.model.input_dim}) != dataset input dim ({self.dataset.input_dim})")
        
        # Ensure sufficient clients for federated learning
        if self.dataset.num_clients < self.federated.min_available_clients:
            raise ConfigurationError(
                f"Dataset num_clients ({self.dataset.num_clients}) < "
                f"federated min_available_clients ({self.federated.min_available_clients})"
            )
    
    def _setup_output_directory(self) -> None:
        """Setup output directory structure."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'metrics', 'logs', 'plots']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        return {
            'experiment_name': self.experiment_name,
            'dataset': self.dataset.get_summary(),
            'model_architecture': self.model.architecture.value if hasattr(self.model, 'architecture') else 'unknown',
            'trust_mode': self.trust.trust_mode.value,
            'federated_rounds': self.federated.num_rounds,
            'local_epochs': self.training.local_epochs,
            'num_clients': self.dataset.num_clients,
            'output_dir': str(self.output_dir)
        }
