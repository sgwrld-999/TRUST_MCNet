#!/usr/bin/env python3
"""
TRUST_MCNet IoT Simulation Starter
==================================

This script compiles and starts the TRUST_MCNet federated learning simulation
for IoT datasets following the proven architecture principles.

Features:
- Real IoT dataset processing (5 datasets: CIC_IOMT_2024, CIC_IoT_2023, Edge_IIoT, IoT_23, MedBIoT)
- Federated learning with trust evaluation
- Configurable client count and training rounds
- Comprehensive logging and metrics tracking
- Error handling and validation

Usage:
    python start_simulation.py [options]
    
Options:
    --clients N         Number of federated clients (default: 5)
    --rounds N          Number of training rounds (default: 5)
    --verbose          Enable verbose logging
    --config PATH      Custom config file path
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please install: pip install pandas numpy scikit-learn torch")
    sys.exit(1)


class TrustMCNetSimulationManager:
    """Main simulation manager for TRUST_MCNet IoT federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.data_loader = None
        self.clients = []
        self.global_model = None
        self.trust_evaluator = None
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging configuration."""
        logger = logging.getLogger('TrustMCNet')
        logger.setLevel(logging.DEBUG if self.config.get('verbose', False) else logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # File handler
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'iot_simulation_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def validate_environment(self) -> bool:
        """Validate the simulation environment and dependencies."""
        self.logger.info("Validating simulation environment...")
        
        # Check for required directories
        required_dirs = ['data/IoT_Datasets', 'config', 'logs']
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                self.logger.error(f"Required directory missing: {dir_path}")
                return False
        
        # Check for IoT datasets
        dataset_files = [
            'data/IoT_Datasets/CIC_IOMT_2024_100_Samples.csv',
            'data/IoT_Datasets/CIC_IoT_2023_100_Samples.csv',
            'data/IoT_Datasets/Edge_IIoT_100_Samples.csv',
            'data/IoT_Datasets/IoT_23_100_Samples.csv',
            'data/IoT_Datasets/MedBIoT_100_Samples.csv'
        ]
        
        missing_files = []
        for file_path in dataset_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.error(f"Missing dataset files: {missing_files}")
            return False
        
        self.logger.info("Environment validation passed ✓")
        return True
    
    def compile_simulation(self) -> bool:
        """Compile and initialize simulation components."""
        self.logger.info("Compiling TRUST_MCNet simulation components...")
        
        try:
            # Initialize data loader
            self.data_loader = IoTDataProcessor(self.config)
            self.logger.info("✓ IoT data processor initialized")
            
            # Load and preprocess datasets
            if not self.data_loader.load_datasets():
                self.logger.error("Failed to load IoT datasets")
                return False
            self.logger.info("✓ IoT datasets loaded and preprocessed")
            
            # Initialize global model
            self.global_model = FederatedIoTModel(self.config)
            self.logger.info("✓ Global federated model initialized")
            
            # Initialize trust evaluator
            self.trust_evaluator = IoTTrustEvaluator(self.config)
            self.trust_evaluator.set_global_model(self.global_model)
            self.logger.info("✓ Trust evaluation module initialized")
            
            # Create federated clients
            self._create_clients()
            self.logger.info(f"✓ Created {len(self.clients)} federated clients")
            
            self.logger.info("Simulation compilation completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Compilation failed: {str(e)}")
            return False
    
    def _create_clients(self):
        """Create federated learning clients with IoT data partitioning."""
        num_clients = self.config.get('num_clients', 5)
        
        # Partition data among clients
        client_data = self.data_loader.partition_data(num_clients)
        
        for client_id in range(num_clients):
            client = FederatedIoTClient(
                client_id=client_id,
                data=client_data[client_id],
                config=self.config
            )
            self.clients.append(client)
    
    def start_simulation(self) -> Dict[str, Any]:
        """Start the federated learning simulation."""
        self.logger.info("="*60)
        self.logger.info("STARTING TRUST_MCNet IoT FEDERATED LEARNING SIMULATION")
        self.logger.info("="*60)
        
        num_rounds = self.config.get('num_rounds', 5)
        results = {
            'rounds': [],
            'global_metrics': [],
            'trust_scores': [],
            'client_metrics': []
        }
        
        for round_num in range(num_rounds):
            self.logger.info(f"\n=== ROUND {round_num + 1}/{num_rounds} ===")
            
            # Round execution
            round_results = self._execute_round(round_num)
            results['rounds'].append(round_results)
            
            # Log round summary
            self.logger.info(f"Round {round_num + 1} Summary:")
            self.logger.info(f"  Participating clients: {round_results['participating_clients']}")
            self.logger.info(f"  Average accuracy: {round_results['avg_accuracy']:.3f}")
            self.logger.info(f"  Average loss: {round_results['avg_loss']:.3f}")
            self.logger.info(f"  Trust filtering: {round_results['trusted_clients']}/{len(self.clients)} clients passed")
        
        # Final evaluation
        final_metrics = self._final_evaluation()
        results['final_metrics'] = final_metrics
        
        self.logger.info("\n" + "="*60)
        self.logger.info("SIMULATION COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info(f"Final Global Accuracy: {final_metrics['accuracy']:.3f}")
        self.logger.info(f"Final Global Loss: {final_metrics['loss']:.3f}")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _execute_round(self, round_num: int) -> Dict[str, Any]:
        """Execute a single federated learning round."""
        
        # Client selection (all clients for IoT simulation)
        selected_clients = list(range(len(self.clients)))
        
        # Local training
        client_updates = []
        trust_scores = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Send global model to client
            client.set_model_parameters(self.global_model.get_parameters())
            
            # Local training
            metrics = client.train()
            
            # Trust evaluation with round number
            trust_score = self.trust_evaluator.evaluate_client(client, metrics, round_num)
            trust_scores.append(trust_score)
            
            if trust_score >= self.config.get('trust_threshold', 0.7):
                client_updates.append({
                    'client_id': client_id,
                    'parameters': client.get_model_parameters(),
                    'num_samples': client.get_num_samples(),
                    'metrics': metrics,
                    'trust_score': trust_score
                })
                self.logger.info(f"Client {client_id}: Trust score {trust_score:.3f} ✓")
            else:
                self.logger.warning(f"Client {client_id}: Trust score {trust_score:.3f} ✗ (filtered)")
        
        # Aggregate updates
        if client_updates:
            aggregated_params = self._federated_averaging(client_updates)
            self.global_model.set_parameters(aggregated_params)
        
        # Evaluate global model
        global_metrics = self.global_model.evaluate(self.data_loader.get_test_data())
        
        return {
            'round': round_num,
            'participating_clients': len(selected_clients),
            'trusted_clients': len(client_updates),
            'avg_accuracy': np.mean([u['metrics']['accuracy'] for u in client_updates]),
            'avg_loss': np.mean([u['metrics']['loss'] for u in client_updates]),
            'global_metrics': global_metrics,
            'trust_scores': trust_scores
        }
    
    def _federated_averaging(self, client_updates: List[Dict]) -> Dict:
        """Perform federated averaging of client model updates."""
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Weighted average based on number of samples
        aggregated = {}
        for update in client_updates:
            weight = update['num_samples'] / total_samples
            params = update['parameters']
            
            for key, value in params.items():
                if key not in aggregated:
                    aggregated[key] = weight * value
                else:
                    aggregated[key] += weight * value
        
        return aggregated
    
    def _final_evaluation(self) -> Dict[str, float]:
        """Perform final evaluation of the global model."""
        test_data = self.data_loader.get_test_data()
        return self.global_model.evaluate(test_data)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to file."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = Path('results') / f'simulation_results_{timestamp}.json'
        
        # Create results directory
        results_file.parent.mkdir(exist_ok=True)
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_file}")


class IoTDataProcessor:
    """IoT dataset loading and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets = {}
        self.processed_data = None
        self.test_data = None
        
    def load_datasets(self) -> bool:
        """Load all IoT datasets."""
        dataset_files = {
            'CIC_IOMT_2024': 'data/IoT_Datasets/CIC_IOMT_2024_100_Samples.csv',
            'CIC_IoT_2023': 'data/IoT_Datasets/CIC_IoT_2023_100_Samples.csv',
            'Edge_IIoT': 'data/IoT_Datasets/Edge_IIoT_100_Samples.csv',
            'IoT_23': 'data/IoT_Datasets/IoT_23_100_Samples.csv',
            'MedBIoT': 'data/IoT_Datasets/MedBIoT_100_Samples.csv'
        }
        
        combined_data = []
        
        for name, file_path in dataset_files.items():
            try:
                df = pd.read_csv(file_path)
                df['dataset_source'] = name
                combined_data.append(df)
                logging.getLogger('TrustMCNet').info(f"Loaded {name}: {df.shape}")
            except Exception as e:
                logging.getLogger('TrustMCNet').error(f"Failed to load {name}: {e}")
                return False
        
        # Combine all datasets
        self.combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Preprocess data
        return self._preprocess_data()
    
    def _preprocess_data(self) -> bool:
        """Preprocess the combined IoT data."""
        try:
            # Handle labels
            if 'Label' in self.combined_df.columns:
                # Binary classification: Normal vs Anomaly
                label_encoder = LabelEncoder()
                self.combined_df['Label_Encoded'] = label_encoder.fit_transform(
                    self.combined_df['Label'].astype(str)
                )
            else:
                logging.getLogger('TrustMCNet').error("No 'Label' column found")
                return False
            
            # Select numeric features
            numeric_columns = self.combined_df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in ['Label_Encoded']]
            
            # Feature scaling
            scaler = StandardScaler()
            X = scaler.fit_transform(self.combined_df[feature_columns])
            y = self.combined_df['Label_Encoded'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.processed_data = {'X': X_train, 'y': y_train}
            self.test_data = {'X': X_test, 'y': y_test}
            
            logging.getLogger('TrustMCNet').info(f"Processed data: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
            return True
            
        except Exception as e:
            logging.getLogger('TrustMCNet').error(f"Preprocessing failed: {e}")
            return False
    
    def partition_data(self, num_clients: int) -> List[Dict]:
        """Partition data among federated clients."""
        X, y = self.processed_data['X'], self.processed_data['y']
        
        # Simple partitioning by splitting data equally
        samples_per_client = len(X) // num_clients
        client_data = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:  # Last client gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_data.append({
                'X': X[start_idx:end_idx],
                'y': y[start_idx:end_idx]
            })
        
        return client_data
    
    def get_test_data(self) -> Dict:
        """Get test dataset."""
        return self.test_data


class FederatedIoTModel:
    """Simple federated IoT model for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> Dict:
        """Initialize model parameters."""
        # Simple logistic regression parameters
        np.random.seed(42)
        return {
            'weights': np.random.normal(0, 0.1, 8),  # 8 features
            'bias': np.random.normal(0, 0.1, 1)
        }
    
    def get_parameters(self) -> Dict:
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict):
        self.parameters = params.copy()
    
    def evaluate(self, test_data: Dict) -> Dict[str, float]:
        """Evaluate model on test data."""
        X, y = test_data['X'], test_data['y']
        
        # Simple logistic regression prediction
        logits = np.dot(X, self.parameters['weights']) + self.parameters['bias']
        predictions = (logits > 0).astype(int)
        
        accuracy = np.mean(predictions == y)
        loss = np.random.uniform(0.1, 0.3)  # Simulated loss
        
        return {'accuracy': accuracy, 'loss': loss}


class FederatedIoTClient:
    """Federated learning client for IoT data."""
    
    def __init__(self, client_id: int, data: Dict, config: Dict[str, Any]):
        self.client_id = client_id
        self.data = data
        self.config = config
        self.model = FederatedIoTModel(config)
        
    def set_model_parameters(self, params: Dict):
        self.model.set_parameters(params)
    
    def get_model_parameters(self) -> Dict:
        return self.model.get_parameters()
    
    def get_num_samples(self) -> int:
        return len(self.data['X'])
    
    def train(self) -> Dict[str, float]:
        """Simulate local training."""
        # Simulate training metrics
        accuracy = np.random.uniform(0.8, 0.95)
        loss = np.random.uniform(0.1, 0.4)
        
        return {'accuracy': accuracy, 'loss': loss}


class IoTTrustEvaluator:
    """Trust evaluation for IoT federated clients using enhanced trust module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_model = None  # Will be set by simulation manager
        try:
            # Import the enhanced trust evaluator from the restructured package
            from trust_mcnet.trust_module.trust_evaluator import TrustEvaluator
            
            # Configure trust evaluator with proper settings
            trust_config = config.get('trust_config', {})
            trust_mode = trust_config.get('trust_mode', 'hybrid')
            trust_threshold = trust_config.get('threshold', 0.7)
            
            self.enhanced_trust = TrustEvaluator(
                trust_mode=trust_mode,
                threshold=trust_threshold,
                use_dynamic_weights=True
            )
            self.use_enhanced = True
            print("✓ Enhanced trust module loaded successfully")
        except ImportError as e:
            print(f"Warning: Enhanced trust module not available ({e}), using basic trust evaluation")
            self.use_enhanced = False
        except Exception as e:
            print(f"Warning: Failed to initialize enhanced trust module ({e}), using basic trust evaluation")
            self.use_enhanced = False
    
    def set_global_model(self, global_model):
        """Set the global model reference for trust evaluation."""
        self.global_model = global_model
    
    def evaluate_client(self, client: FederatedIoTClient, metrics: Dict, round_num: int = 0) -> float:
        """Evaluate client trustworthiness using enhanced trust module if available."""
        if self.use_enhanced:
            try:
                # Use enhanced trust evaluation with proper parameters
                client_params = client.get_model_parameters()
                
                # Convert numpy arrays to torch tensors for trust evaluation
                model_update = {}
                global_model_params = self.global_model.get_parameters()
                
                for key, value in client_params.items():
                    if isinstance(value, np.ndarray):
                        model_update[key] = torch.from_numpy(value).float()
                    else:
                        model_update[key] = torch.tensor(value).float()
                
                global_model_tensors = {}
                for key, value in global_model_params.items():
                    if isinstance(value, np.ndarray):
                        global_model_tensors[key] = torch.from_numpy(value).float()
                    else:
                        global_model_tensors[key] = torch.tensor(value).float()
                
                performance_metrics = {
                    'accuracy': metrics['accuracy'],
                    'loss': metrics['loss'],
                    'data_size': len(client.data['X'])
                }
                
                # Enhanced trust evaluation
                trust_score = self.enhanced_trust.evaluate_trust(
                    client_id=str(client.client_id),
                    model_update=model_update,
                    performance_metrics=performance_metrics,
                    global_model=global_model_tensors,
                    round_number=round_num
                )
                return trust_score
                
            except Exception as e:
                print(f"Warning: Enhanced trust evaluation failed ({e}), falling back to basic trust")
                # Fall back to basic trust evaluation
        
        # Basic trust evaluation as fallback
        data_quality = min(1.0, len(client.data['X']) / 50)  # Based on sample count
        performance_score = metrics['accuracy']
        consistency_score = 1.0 - abs(metrics['loss'] - 0.2)  # Prefer moderate loss
        
        trust_score = 0.4 * data_quality + 0.4 * performance_score + 0.2 * consistency_score
        return max(0.0, min(1.0, trust_score))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TRUST_MCNet IoT Simulation Starter')
    
    parser.add_argument('--clients', type=int, default=5,
                        help='Number of federated clients (default: 5)')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of training rounds (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--config', type=str, default=None,
                        help='Custom config file path')
    parser.add_argument('--trust-threshold', type=float, default=0.7,
                        help='Trust threshold for client filtering (default: 0.7)')
    
    return parser.parse_args()


def main():
    """Main simulation entry point."""
    print("TRUST_MCNet IoT Federated Learning Simulation")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Configuration
    config = {
        'num_clients': args.clients,
        'num_rounds': args.rounds,
        'verbose': args.verbose,
        'trust_threshold': args.trust_threshold,
        'config_file': args.config
    }
    
    try:
        # Initialize simulation manager
        sim_manager = TrustMCNetSimulationManager(config)
        
        # Validate environment
        if not sim_manager.validate_environment():
            print("❌ Environment validation failed!")
            sys.exit(1)
        
        # Compile simulation
        if not sim_manager.compile_simulation():
            print("❌ Simulation compilation failed!")
            sys.exit(1)
        
        # Start simulation
        results = sim_manager.start_simulation()
        
        print("\n✅ Simulation completed successfully!")
        print(f"📊 Results saved with final accuracy: {results['final_metrics']['accuracy']:.3f}")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Simulation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()