#!/usr/bin/env python3
"""
Simple demonstration of the refactored TRUST-MCNet architecture.
This version uses direct imports and demonstrates the key patterns without complex dependencies.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo.log')
    ]
)
logger = logging.getLogger(__name__)


class MockDataLoader:
    """Mock data loader for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Created MockDataLoader with config: {config}")
    
    def load_data(self):
        logger.info("Loading mock training data...")
        return {"train_data": "mock_data", "test_data": "mock_data"}
    
    def get_client_data(self, client_id: int):
        logger.info(f"Getting data for client {client_id}")
        return {"client_data": f"mock_data_client_{client_id}"}


class MockModel:
    """Mock model for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Created MockModel with config: {config}")
    
    def get_parameters(self):
        logger.info("Getting model parameters")
        return {"weights": "mock_weights"}
    
    def set_parameters(self, parameters):
        logger.info("Setting model parameters")
    
    def train(self, data):
        logger.info("Training model on data")
        return {"loss": 0.1, "accuracy": 0.95}
    
    def evaluate(self, data):
        logger.info("Evaluating model on data")
        return {"test_loss": 0.05, "test_accuracy": 0.98}


class MockTrustEvaluator:
    """Mock trust evaluator for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Created MockTrustEvaluator with config: {config}")
    
    def evaluate_client_trust(self, client_id: int, update_data: Dict) -> float:
        # Simple mock trust score
        trust_score = max(0.5, 1.0 - (client_id * 0.1))
        logger.info(f"Client {client_id} trust score: {trust_score:.3f}")
        return trust_score
    
    def filter_trusted_clients(self, client_updates: List[Dict]) -> List[Dict]:
        """Filter clients based on trust scores."""
        trusted_updates = []
        for update in client_updates:
            client_id = update.get('client_id', 0)
            trust_score = self.evaluate_client_trust(client_id, update)
            if trust_score > 0.6:  # Trust threshold
                trusted_updates.append(update)
                logger.info(f"Client {client_id} passed trust filter")
            else:
                logger.warning(f"Client {client_id} failed trust filter")
        return trusted_updates


class MockFederatedStrategy:
    """Mock federated learning strategy for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"Created MockFederatedStrategy with config: {config}")
    
    def aggregate_updates(self, client_updates: List[Dict]) -> Dict:
        logger.info(f"Aggregating updates from {len(client_updates)} clients")
        # Simple mock aggregation
        return {"aggregated_weights": "mock_aggregated_weights"}
    
    def select_clients(self, available_clients: List[int], round_num: int) -> List[int]:
        # Simple client selection
        num_clients = min(self.config.get('clients_per_round', 3), len(available_clients))
        selected = available_clients[:num_clients]
        logger.info(f"Round {round_num}: Selected clients {selected}")
        return selected


class MockClient:
    """Mock federated learning client for demonstration."""
    
    def __init__(self, client_id: int, model: MockModel, data_loader: MockDataLoader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        logger.info(f"Created MockClient {client_id}")
    
    def train_local_model(self, global_parameters: Dict) -> Dict:
        logger.info(f"Client {self.client_id}: Starting local training")
        
        # Set global parameters
        self.model.set_parameters(global_parameters)
        
        # Get local data
        local_data = self.data_loader.get_client_data(self.client_id)
        
        # Train locally
        training_results = self.model.train(local_data)
        
        # Return update
        update = {
            'client_id': self.client_id,
            'parameters': self.model.get_parameters(),
            'metrics': training_results,
            'data_size': 100  # Mock data size
        }
        
        logger.info(f"Client {self.client_id}: Local training complete")
        return update


class MockExperimentManager:
    """Mock experiment manager demonstrating the architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing MockExperimentManager")
        
        # Initialize components
        self.data_loader = MockDataLoader(config.get('dataset', {}))
        self.global_model = MockModel(config.get('model', {}))
        self.trust_evaluator = MockTrustEvaluator(config.get('trust', {}))
        self.strategy = MockFederatedStrategy(config.get('strategy', {}))
        
        # Initialize clients
        num_clients = config.get('num_clients', 5)
        self.clients = []
        for i in range(num_clients):
            client_model = MockModel(config.get('model', {}))
            client = MockClient(i, client_model, self.data_loader)
            self.clients.append(client)
        
        logger.info(f"Created {len(self.clients)} clients")
    
    def run_experiment(self):
        """Run the federated learning experiment."""
        logger.info("Starting federated learning experiment")
        
        try:
            # Load global data for evaluation
            global_data = self.data_loader.load_data()
            
            # Get initial global parameters
            global_parameters = self.global_model.get_parameters()
            
            # Run federated learning rounds
            num_rounds = self.config.get('num_rounds', 3)
            for round_num in range(num_rounds):
                logger.info(f"\n=== Round {round_num + 1}/{num_rounds} ===")
                
                # Select clients for this round
                available_clients = list(range(len(self.clients)))
                selected_client_ids = self.strategy.select_clients(available_clients, round_num)
                
                # Collect client updates
                client_updates = []
                for client_id in selected_client_ids:
                    client = self.clients[client_id]
                    update = client.train_local_model(global_parameters)
                    client_updates.append(update)
                
                # Apply trust filtering
                trusted_updates = self.trust_evaluator.filter_trusted_clients(client_updates)
                logger.info(f"Trust filter: {len(trusted_updates)}/{len(client_updates)} clients passed")
                
                # Aggregate trusted updates
                if trusted_updates:
                    aggregated_update = self.strategy.aggregate_updates(trusted_updates)
                    global_parameters = aggregated_update
                    logger.info("Global model updated with aggregated parameters")
                else:
                    logger.warning("No trusted updates - skipping aggregation")
                
                # Evaluate global model
                evaluation_results = self.global_model.evaluate(global_data)
                logger.info(f"Round {round_num + 1} Global Model Performance: {evaluation_results}")
            
            logger.info("Federated learning experiment completed successfully!")
            return {"status": "success", "final_metrics": evaluation_results}
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using default config")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration, with environment variable overrides."""
    # Get configuration from environment variables if available
    dataset_name = os.environ.get('EXPERIMENT_DATASET', 'mock_dataset')
    num_clients = int(os.environ.get('EXPERIMENT_CLIENTS', '5'))
    num_rounds = int(os.environ.get('EXPERIMENT_ROUNDS', '3'))
    experiment_id = os.environ.get('EXPERIMENT_ID', 'default')
    
    # Determine clients per round (max 3 or total clients if less)
    clients_per_round = min(3, num_clients)
    
    logger.info(f"Configuration override from environment:")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Clients: {num_clients}")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  Experiment ID: {experiment_id}")
    
    return {
        'experiment_id': experiment_id,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'dataset': {
            'name': dataset_name,
            'path': './data'
        },
        'model': {
            'type': 'mock_model',
            'input_dim': 784,
            'num_classes': 10
        },
        'strategy': {
            'name': 'fedavg',
            'clients_per_round': clients_per_round
        },
        'trust': {
            'type': 'cosine_similarity',
            'threshold': 0.6
        }
    }


def main():
    """Main entry point for the demonstration."""
    parser = argparse.ArgumentParser(description="TRUST-MCNet Refactored Architecture Demo")
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting TRUST-MCNet Refactored Architecture Demonstration")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration: {config}")
        
        # Create and run experiment
        experiment_manager = MockExperimentManager(config)
        results = experiment_manager.run_experiment()
        
        logger.info("=" * 60)
        logger.info("Demonstration completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
