"""
Baseline federated learning script for TRUST-MCNet with random weight aggregation.

This script implements a baseline federated learning pipeline using MNIST dataset
with random weight aggregation for pipeline verification and comparison.
"""

import argparse
import logging
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import ConfigManager, MNISTDataLoader
from models.model import MLP, LSTM
from clients.client import Client
from utils.results_manager import (
    ExperimentLogger, ResultsVisualizer, calculate_detection_metrics
)


class RandomWeightFederatedServer:
    """
    Federated server with random weight aggregation for baseline testing.
    """
    
    def __init__(self, global_model: nn.Module, config: Dict[str, Any], logger: ExperimentLogger):
        """
        Initialize federated server with random aggregation.
        
        Args:
            global_model: Global model architecture
            config: Configuration dictionary
            logger: Experiment logger
        """
        self.global_model = global_model
        self.config = config
        self.logger = logger
        self.round_number = 0
        
        # Federated learning parameters
        self.num_rounds = config.get('federated', {}).get('num_rounds', 50)
        self.num_clients = config.get('federated', {}).get('num_clients', 10)
        self.min_available_clients = config.get('federated', {}).get('min_available_clients', 5)
        
        # Random weight configuration
        self.baseline_mode = config.get('experiment', {}).get('baseline_mode', True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        # Setup logging
        self.fed_logger = logging.getLogger(__name__)
        
        self.fed_logger.info(f"Initialized federated server with random aggregation")
        self.fed_logger.info(f"Device: {self.device}")
        self.fed_logger.info(f"Baseline mode: {self.baseline_mode}")
    
    def generate_random_weights(self, num_clients: int) -> Dict[str, float]:
        """
        Generate random weights for client aggregation.
        
        Args:
            num_clients: Number of participating clients
            
        Returns:
            Dictionary mapping client IDs to weights
        """
        # Generate random weights that sum to 1
        raw_weights = np.random.exponential(scale=1.0, size=num_clients)
        normalized_weights = raw_weights / np.sum(raw_weights)
        
        # Create client weight mapping
        weights = {}
        for i in range(num_clients):
            weights[f"client_{i}"] = float(normalized_weights[i])
        
        self.fed_logger.info(f"Generated random weights: {weights}")
        return weights
    
    def aggregate_models(self, client_models: List[Dict[str, torch.Tensor]], 
                        client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using random weights.
        
        Args:
            client_models: List of client model state dictionaries
            client_weights: Dictionary of client weights
            
        Returns:
            Aggregated model state dictionary
        """
        if not client_models:
            raise ValueError("No client models provided for aggregation")
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first model
        param_names = client_models[0].keys()
        
        for param_name in param_names:
            # Initialize with zeros
            aggregated_params[param_name] = torch.zeros_like(client_models[0][param_name])
            
            # Weighted sum
            for i, client_model in enumerate(client_models):
                client_id = f"client_{i}"
                weight = client_weights.get(client_id, 1.0 / len(client_models))
                aggregated_params[param_name] += weight * client_model[param_name]
        
        self.fed_logger.info(f"Aggregated models from {len(client_models)} clients")
        return aggregated_params
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the global model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.global_model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        metrics = calculate_detection_metrics(np.array(all_labels), np.array(all_predictions))
        metrics['loss'] = avg_loss
        
        self.fed_logger.info(f"Global model evaluation: {metrics}")
        return metrics
    
    def run_federated_training(self, clients: List[Client], test_loader: DataLoader) -> Dict[str, Any]:
        """
        Run federated training with random weight aggregation.
        
        Args:
            clients: List of federated clients
            test_loader: Test data loader for evaluation
            
        Returns:
            Training results and metrics
        """
        self.logger.log_training_event("training_started", {
            "num_clients": len(clients),
            "num_rounds": self.num_rounds,
            "baseline_mode": self.baseline_mode
        })
        
        training_results = {
            "round_metrics": [],
            "final_metrics": {},
            "client_weights_history": []
        }
        
        for round_num in range(1, self.num_rounds + 1):
            self.round_number = round_num
            self.fed_logger.info(f"Starting round {round_num}/{self.num_rounds}")
            
            try:
                # Select clients (use all available for baseline)
                participating_clients = clients[:min(len(clients), self.num_clients)]
                
                if len(participating_clients) < self.min_available_clients:
                    self.fed_logger.warning(f"Insufficient clients: {len(participating_clients)} < {self.min_available_clients}")
                    continue
                
                # Generate random weights for this round
                client_weights = self.generate_random_weights(len(participating_clients))
                
                # Log client weights
                self.logger.log_client_weights(round_num, client_weights)
                training_results["client_weights_history"].append({
                    "round": round_num,
                    "weights": client_weights
                })
                
                # Distribute global model to clients
                global_params = self.global_model.state_dict()
                for client in participating_clients:
                    client.set_weights(global_params)
                
                # Client training
                client_models = []
                client_metrics = []
                
                for i, client in enumerate(participating_clients):
                    client_id = f"client_{i}"
                    self.fed_logger.info(f"Training {client_id}")
                    
                    try:
                        # Train client
                        local_metrics = client.train()
                        client_models.append(client.get_weights())
                        client_metrics.append(local_metrics)
                        
                        self.logger.log_training_event("client_training_completed", {
                            "client_id": client_id,
                            "round": round_num,
                            "metrics": local_metrics
                        })
                        
                    except Exception as e:
                        self.fed_logger.error(f"Client {client_id} training failed: {e}")
                        self.logger.log_training_event("client_training_failed", {
                            "client_id": client_id,
                            "round": round_num,
                            "error": str(e)
                        })
                        continue
                
                if not client_models:
                    self.fed_logger.error(f"No successful client training in round {round_num}")
                    continue
                
                # Aggregate models with random weights
                try:
                    aggregated_params = self.aggregate_models(client_models, client_weights)
                    self.global_model.load_state_dict(aggregated_params)
                    
                    self.logger.log_training_event("aggregation_completed", {
                        "round": round_num,
                        "num_clients_aggregated": len(client_models)
                    })
                    
                except Exception as e:
                    self.fed_logger.error(f"Model aggregation failed in round {round_num}: {e}")
                    continue
                
                # Evaluate global model
                try:
                    global_metrics = self.evaluate_global_model(test_loader)
                    
                    # Combine round metrics
                    round_metrics = {
                        "round": round_num,
                        "num_participating_clients": len(participating_clients),
                        "num_successful_clients": len(client_models),
                        "train_accuracy": np.mean([m.get('accuracy', 0) for m in client_metrics]),
                        "train_loss": np.mean([m.get('loss', 0) for m in client_metrics]),
                        "test_accuracy": global_metrics.get('accuracy', 0),
                        "test_loss": global_metrics.get('loss', 0),
                        "test_f1_score": global_metrics.get('f1_score', 0),
                        "test_precision": global_metrics.get('precision', 0),
                        "test_recall": global_metrics.get('recall', 0),
                        "detection_rate": global_metrics.get('detection_rate', 0)
                    }
                    
                    # Log round metrics
                    self.logger.log_round_metrics(round_num, round_metrics)
                    training_results["round_metrics"].append(round_metrics)
                    
                    self.fed_logger.info(f"Round {round_num} completed. Test accuracy: {global_metrics.get('accuracy', 0):.4f}")
                    
                except Exception as e:
                    self.fed_logger.error(f"Global model evaluation failed in round {round_num}: {e}")
                    continue
                    
            except Exception as e:
                self.fed_logger.error(f"Round {round_num} failed: {e}")
                self.fed_logger.error(traceback.format_exc())
                continue
        
        # Final evaluation
        try:
            final_metrics = self.evaluate_global_model(test_loader)
            training_results["final_metrics"] = final_metrics
            
            self.logger.log_training_event("training_completed", {
                "total_rounds": self.num_rounds,
                "final_metrics": final_metrics
            })
            
        except Exception as e:
            self.fed_logger.error(f"Final evaluation failed: {e}")
        
        return training_results


def train_centralized_baseline(model: nn.Module, train_loader: DataLoader, 
                              val_loader: DataLoader, test_loader: DataLoader,
                              config: Dict[str, Any], logger: ExperimentLogger) -> Dict[str, float]:
    """
    Train a centralized model for baseline comparison.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary
        logger: Experiment logger
        
    Returns:
        Dictionary of final metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training configuration
    learning_rate = config.get('client', {}).get('learning_rate', 0.001)
    epochs = config.get('training', {}).get('epochs', 20)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    cent_logger = logging.getLogger(f"{__name__}.centralized")
    cent_logger.info(f"Training centralized baseline model")
    cent_logger.info(f"Device: {device}, Epochs: {epochs}, LR: {learning_rate}")
    
    logger.log_training_event("centralized_training_started", {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": str(device)
    })
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        cent_logger.info(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Final test evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final metrics
    final_metrics = calculate_detection_metrics(np.array(all_labels), np.array(all_predictions))
    final_metrics['loss'] = test_loss / len(test_loader)
    
    logger.log_training_event("centralized_training_completed", {
        "final_metrics": final_metrics
    })
    
    cent_logger.info(f"Centralized training completed. Test accuracy: {final_metrics.get('accuracy', 0):.4f}")
    
    return final_metrics


def create_federated_clients(data_loader: MNISTDataLoader, config: Dict[str, Any]) -> List[Client]:
    """
    Create federated learning clients with MNIST data.
    
    Args:
        data_loader: MNIST data loader
        config: Configuration dictionary
        
    Returns:
        List of federated clients
    """
    num_clients = config.get('federated', {}).get('num_clients', 10)
    distribution = config.get('data', {}).get('distribution', 'iid')
    
    # Create federated datasets
    client_datasets = data_loader.create_federated_datasets(
        num_clients=num_clients,
        distribution=distribution
    )
    
    # Create clients
    clients = []
    model_config = config.get('model', {})
    
    for i, (train_dataset, test_dataset) in enumerate(client_datasets):
        # Create model for client
        if model_config.get('type') == 'MLP':
            client_model = MLP(
                input_dim=model_config.get('mlp', {}).get('input_dim', 784),
                hidden_dims=model_config.get('mlp', {}).get('hidden_dims', [256, 128, 64]),
                output_dim=model_config.get('mlp', {}).get('output_dim', 2)
            )
        elif model_config.get('type') == 'LSTM':
            client_model = LSTM(
                input_dim=model_config.get('lstm', {}).get('input_dim', 784),
                hidden_dim=model_config.get('lstm', {}).get('hidden_dim', 128),
                num_layers=model_config.get('lstm', {}).get('num_layers', 2),
                output_dim=model_config.get('lstm', {}).get('output_dim', 2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.get('type')}")
        
        # Create client
        client = Client(
            client_id=f"client_{i}",
            model=client_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config
        )
        
        clients.append(client)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created {len(clients)} federated clients")
    
    return clients


def main():
    """Main function for running baseline federated learning experiment."""
    parser = argparse.ArgumentParser(
        description="TRUST-MCNet Baseline Federated Learning with Random Weights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mnist_baseline",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Number of federated rounds (overrides config)"
    )
    parser.add_argument(
        "--clients",
        type=int,
        help="Number of clients (overrides config)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.cfg
        
        # Override with command line arguments
        if args.rounds:
            config.setdefault('federated', {})['num_rounds'] = args.rounds
        if args.clients:
            config.setdefault('federated', {})['num_clients'] = args.clients
        
        # Initialize experiment logger
        experiment_logger = ExperimentLogger(
            experiment_name=args.experiment_name,
            results_dir=config.get('experiment', {}).get('results_dir', 'results')
        )
        
        # Log experiment configuration
        experiment_logger.log_experiment_config(config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        main_logger = logging.getLogger(__name__)
        main_logger.info(f"Starting TRUST-MCNet baseline experiment: {args.experiment_name}")
        
        # Initialize MNIST data loader
        mnist_loader = MNISTDataLoader(config_manager)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = mnist_loader.prepare_datasets()
        
        # Log dataset statistics
        dataset_stats = mnist_loader._log_dataset_statistics()
        experiment_logger.log_dataset_stats(dataset_stats)
        
        # Create global model
        model_config = config.get('model', {})
        if model_config.get('type') == 'MLP':
            global_model = MLP(
                input_dim=model_config.get('mlp', {}).get('input_dim', 784),
                hidden_dims=model_config.get('mlp', {}).get('hidden_dims', [256, 128, 64]),
                output_dim=model_config.get('mlp', {}).get('output_dim', 2)
            )
        elif model_config.get('type') == 'LSTM':
            global_model = LSTM(
                input_dim=model_config.get('lstm', {}).get('input_dim', 784),
                hidden_dim=model_config.get('lstm', {}).get('hidden_dim', 128),
                num_layers=model_config.get('lstm', {}).get('num_layers', 2),
                output_dim=model_config.get('lstm', {}).get('output_dim', 2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.get('type')}")
        
        main_logger.info(f"Created global model: {model_config.get('type')}")
        
        # Create test data loader
        batch_size = config.get('client', {}).get('batch_size', 32)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Create federated clients
        main_logger.info("Creating federated clients...")
        federated_clients = create_federated_clients(mnist_loader, config)
        
        # Initialize federated server
        fed_server = RandomWeightFederatedServer(global_model, config, experiment_logger)
        
        # Run federated training
        main_logger.info("Starting federated training with random weights...")
        fed_results = fed_server.run_federated_training(federated_clients, test_loader)
        
        # Train centralized baseline
        main_logger.info("Training centralized baseline...")
        train_loader, val_loader, cent_test_loader = mnist_loader.get_centralized_datasets()
        
        # Create separate model for centralized training
        if model_config.get('type') == 'MLP':
            centralized_model = MLP(
                input_dim=model_config.get('mlp', {}).get('input_dim', 784),
                hidden_dims=model_config.get('mlp', {}).get('hidden_dims', [256, 128, 64]),
                output_dim=model_config.get('mlp', {}).get('output_dim', 2)
            )
        else:
            centralized_model = LSTM(
                input_dim=model_config.get('lstm', {}).get('input_dim', 784),
                hidden_dim=model_config.get('lstm', {}).get('hidden_dim', 128),
                num_layers=model_config.get('lstm', {}).get('num_layers', 2),
                output_dim=model_config.get('lstm', {}).get('output_dim', 2)
            )
        
        centralized_metrics = train_centralized_baseline(
            centralized_model, train_loader, val_loader, cent_test_loader, config, experiment_logger
        )
        
        # Combine final metrics
        final_metrics = {
            "federated": fed_results.get("final_metrics", {}),
            "centralized": centralized_metrics
        }
        
        experiment_logger.log_final_metrics(final_metrics)
        
        # Create visualizations
        main_logger.info("Generating visualizations...")
        visualizer = ResultsVisualizer(experiment_logger)
        
        # Plot training curves
        visualizer.plot_training_curves()
        
        # Plot weight distributions
        visualizer.plot_weight_distribution()
        
        # Create metrics summary
        summary_df = visualizer.create_metrics_summary_table()
        main_logger.info(f"Metrics Summary:\n{summary_df}")
        
        # SHAP analysis
        if config.get('experiment', {}).get('shap_analysis', True):
            visualizer.create_shap_analysis(
                global_model, 
                test_loader, 
                num_samples=config.get('experiment', {}).get('shap_samples', 100)
            )
        
        # Save all metrics
        experiment_logger.save_metrics()
        
        main_logger.info(f"Experiment completed successfully!")
        main_logger.info(f"Results saved to: {experiment_logger.get_experiment_dir()}")
        
        # Print final summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Results Directory: {experiment_logger.get_experiment_dir()}")
        print(f"\nFederated Learning (Random Weights):")
        fed_final = fed_results.get("final_metrics", {})
        print(f"  Accuracy: {fed_final.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {fed_final.get('f1_score', 0):.4f}")
        print(f"  Detection Rate: {fed_final.get('detection_rate', 0):.4f}")
        print(f"\nCentralized Baseline:")
        print(f"  Accuracy: {centralized_metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {centralized_metrics.get('f1_score', 0):.4f}")
        print(f"  Detection Rate: {centralized_metrics.get('detection_rate', 0):.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
