"""
FLTrust Configuration Example for TRUST-MCNet

This file demonstrates various configuration options for integrating
FLTrust aggregation with TRUST-MCNet.
"""

# Basic FLTrust Configuration
FLTRUST_CONFIG = {
    # Server settings
    'server_learning_rate': 1.0,           # η - Global learning rate for server updates
    'num_rounds': 50,                      # Number of federated rounds
    'num_clients': 10,                     # Total number of clients
    'min_available_clients': 5,            # Minimum clients needed per round
    
    # FLTrust-specific settings
    'clip_threshold': 10.0,                # Maximum norm for gradient clipping
    'warm_up_rounds': 5,                   # Rounds before trust scoring takes effect
    'fltrust_trust_threshold': 0.0,        # Minimum trust score (values below are clipped to 0)
    
    # Trust evaluation settings (TRUST-MCNet integration)
    'trust_mode': 'hybrid',                # 'cosine', 'entropy', 'reputation', 'hybrid'
    'trust_threshold': 0.5,                # TRUST-MCNet trust threshold
    
    # Root dataset settings
    'root_dataset_size': 100,              # Size of trusted root dataset
    'root_batch_size': 32,                 # Batch size for root dataset processing
    
    # Model configuration
    'model_params': {
        'input_dim': 784,
        'hidden_dim': 128,
        'output_dim': 10
    }
}

# Configuration for different scenarios
SCENARIO_CONFIGS = {
    # High security scenario - conservative settings
    'high_security': {
        **FLTRUST_CONFIG,
        'server_learning_rate': 0.5,       # Lower learning rate for stability
        'clip_threshold': 5.0,              # Stricter gradient clipping
        'warm_up_rounds': 10,               # Longer warm-up period
        'fltrust_trust_threshold': 0.2,     # Higher trust threshold
        'trust_threshold': 0.7,             # Stricter TRUST-MCNet threshold
    },
    
    # Fast convergence scenario - aggressive settings
    'fast_convergence': {
        **FLTRUST_CONFIG,
        'server_learning_rate': 2.0,       # Higher learning rate
        'clip_threshold': 20.0,             # More lenient clipping
        'warm_up_rounds': 2,                # Shorter warm-up
        'fltrust_trust_threshold': 0.0,     # No trust threshold
        'trust_threshold': 0.3,             # More lenient TRUST-MCNet threshold
    },
    
    # Balanced scenario - moderate settings
    'balanced': {
        **FLTRUST_CONFIG,
        'server_learning_rate': 1.0,
        'clip_threshold': 10.0,
        'warm_up_rounds': 5,
        'fltrust_trust_threshold': 0.1,
        'trust_threshold': 0.5,
    }
}

# Example of how to use different aggregation strategies
AGGREGATOR_CONFIGS = {
    'fltrust': {
        'type': 'FLTrustAggregator',
        'params': {
            'server_learning_rate': 1.0,
            'clip_threshold': 10.0,
            'warm_up_rounds': 5,
            'trust_threshold': 0.0
        }
    },
    
    'fedavg': {
        'type': 'FedAvgAggregator',
        'params': {
            'clip_threshold': 10.0
        }
    },
    
    'trimmed_mean': {
        'type': 'TrimmedMeanAggregator',
        'params': {
            'trim_ratio': 0.2,
            'clip_threshold': 5.0
        }
    },
    
    'krum': {
        'type': 'KrumAggregator',
        'params': {
            'num_malicious': 1
        }
    }
}

# Trust score combination strategies
TRUST_COMBINATION_CONFIGS = {
    # Use only FLTrust cosine similarity
    'cosine_only': {
        'use_trust_override': False,
        'combination_method': None
    },
    
    # Use only TRUST-MCNet scores
    'trust_mcnet_only': {
        'use_trust_override': True,
        'combination_method': 'replace'
    },
    
    # Weighted combination of both
    'weighted_combination': {
        'use_trust_override': True,
        'combination_method': 'weighted',
        'trust_mcnet_weight': 0.7,
        'cosine_weight': 0.3
    },
    
    # Use TRUST-MCNet as primary, cosine as fallback
    'fallback': {
        'use_trust_override': True,
        'combination_method': 'fallback',
        'fallback_threshold': 0.1
    }
}
