#!/usr/bin/env python3
"""
Multi-Dataset Federated Learning Simulation CLI
Run simulations across multiple datasets with configurable parameters.

Usage:
    python run_multi_dataset_simulation.py --clients 5 --rounds 3 --datasets mnist iot_general custom_csv
    python run_multi_dataset_simulation.py --help
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import subprocess
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'multi_dataset_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class MultiDatasetSimulation:
    """
    Orchestrates federated learning simulations across multiple datasets.
    """
    
    def __init__(self, num_clients: int, num_rounds: int, datasets: List[str]):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.datasets = datasets
        self.results = {}
        self.start_time = None
        self.total_time = None
        
        # Validate datasets
        self.available_datasets = self._get_available_datasets()
        self._validate_datasets()
        
        logger.info(f"Initialized simulation with {num_clients} clients, {num_rounds} rounds")
        logger.info(f"Datasets to process: {', '.join(datasets)}")
    
    def _get_available_datasets(self) -> List[str]:
        """Get list of available dataset configurations."""
        config_dir = Path("config/dataset")
        if not config_dir.exists():
            logger.error(f"Dataset config directory not found: {config_dir}")
            return []
        
        datasets = []
        for yaml_file in config_dir.glob("*.yaml"):
            dataset_name = yaml_file.stem
            datasets.append(dataset_name)
        
        logger.info(f"Available datasets: {', '.join(datasets)}")
        return datasets
    
    def _validate_datasets(self):
        """Validate that all requested datasets are available."""
        invalid_datasets = []
        for dataset in self.datasets:
            if dataset not in self.available_datasets:
                invalid_datasets.append(dataset)
        
        if invalid_datasets:
            logger.error(f"Invalid datasets: {', '.join(invalid_datasets)}")
            logger.error(f"Available datasets: {', '.join(self.available_datasets)}")
            raise ValueError(f"Invalid datasets: {invalid_datasets}")
    
    def run_single_experiment(self, dataset: str, experiment_id: str) -> Dict[str, Any]:
        """
        Run a single federated learning experiment for one dataset.
        """
        logger.info(f"Starting experiment {experiment_id} with dataset '{dataset}'")
        
        # Prepare command
        cmd = [
            "python", "demo_refactored.py",
            "--verbose"
        ]
        
        # Set environment variables for configuration
        env = os.environ.copy()
        env.update({
            'EXPERIMENT_DATASET': dataset,
            'EXPERIMENT_CLIENTS': str(self.num_clients),
            'EXPERIMENT_ROUNDS': str(self.num_rounds),
            'EXPERIMENT_ID': experiment_id
        })
        
        experiment_start = time.time()
        
        try:
            # Run the experiment
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info(f"Environment: Dataset={dataset}, Clients={self.num_clients}, Rounds={self.num_rounds}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            experiment_time = time.time() - experiment_start
            
            if result.returncode == 0:
                logger.info(f"Experiment {experiment_id} completed successfully in {experiment_time:.2f}s")
                
                # Parse results from output
                experiment_results = {
                    'status': 'success',
                    'dataset': dataset,
                    'num_clients': self.num_clients,
                    'num_rounds': self.num_rounds,
                    'execution_time': experiment_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
                # Try to extract metrics from output
                metrics = self._extract_metrics_from_output(result.stdout)
                if metrics:
                    experiment_results['metrics'] = metrics
                
                return experiment_results
            
            else:
                logger.error(f"Experiment {experiment_id} failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                
                return {
                    'status': 'failed',
                    'dataset': dataset,
                    'num_clients': self.num_clients,
                    'num_rounds': self.num_rounds,
                    'execution_time': experiment_time,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'returncode': result.returncode
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Experiment {experiment_id} timed out")
            return {
                'status': 'timeout',
                'dataset': dataset,
                'execution_time': experiment_time,
                'error': 'Experiment timed out after 5 minutes'
            }
        
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed with exception: {str(e)}")
            return {
                'status': 'error',
                'dataset': dataset,
                'error': str(e),
                'execution_time': time.time() - experiment_start
            }
    
    def _extract_metrics_from_output(self, output: str) -> Dict[str, Any]:
        """Extract performance metrics from experiment output."""
        metrics = {}
        
        # Look for specific patterns in the output
        lines = output.split('\n')
        for line in lines:
            if 'test_accuracy' in line.lower():
                try:
                    # Extract accuracy value
                    if ':' in line:
                        accuracy_str = line.split(':')[-1].strip()
                        accuracy_str = accuracy_str.replace(',', '').replace('}', '')
                        accuracy = float(accuracy_str)
                        metrics['final_accuracy'] = accuracy
                except:
                    pass
            
            elif 'test_loss' in line.lower():
                try:
                    # Extract loss value
                    if ':' in line:
                        loss_str = line.split(':')[-1].strip()
                        loss_str = loss_str.replace(',', '').replace('}', '')
                        loss = float(loss_str)
                        metrics['final_loss'] = loss
                except:
                    pass
        
        return metrics
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run federated learning experiments for all specified datasets.
        """
        logger.info("=" * 80)
        logger.info("STARTING MULTI-DATASET FEDERATED LEARNING SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  • Number of clients: {self.num_clients}")
        logger.info(f"  • Number of rounds: {self.num_rounds}")
        logger.info(f"  • Datasets: {', '.join(self.datasets)}")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Run experiments for each dataset
        for i, dataset in enumerate(self.datasets, 1):
            experiment_id = f"exp_{i}_{dataset}_{datetime.now().strftime('%H%M%S')}"
            
            logger.info(f"\n[{i}/{len(self.datasets)}] Processing dataset: {dataset}")
            logger.info("-" * 60)
            
            try:
                result = self.run_single_experiment(dataset, experiment_id)
                self.results[dataset] = result
                
                # Log summary for this dataset
                if result['status'] == 'success':
                    logger.info(f"✅ Dataset '{dataset}' completed successfully")
                    if 'metrics' in result:
                        metrics = result['metrics']
                        logger.info(f"   Final accuracy: {metrics.get('final_accuracy', 'N/A')}")
                        logger.info(f"   Final loss: {metrics.get('final_loss', 'N/A')}")
                    logger.info(f"   Execution time: {result['execution_time']:.2f}s")
                else:
                    logger.error(f"❌ Dataset '{dataset}' failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"❌ Failed to run experiment for dataset '{dataset}': {str(e)}")
                self.results[dataset] = {
                    'status': 'error',
                    'dataset': dataset,
                    'error': str(e)
                }
        
        self.total_time = time.time() - self.start_time
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    def _generate_final_report(self):
        """Generate and display final simulation report."""
        logger.info("\n" + "=" * 80)
        logger.info("SIMULATION COMPLETED - FINAL REPORT")
        logger.info("=" * 80)
        
        successful_experiments = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_experiments = len(self.results) - successful_experiments
        
        logger.info(f"Summary:")
        logger.info(f"  • Total datasets processed: {len(self.datasets)}")
        logger.info(f"  • Successful experiments: {successful_experiments}")
        logger.info(f"  • Failed experiments: {failed_experiments}")
        logger.info(f"  • Total execution time: {self.total_time:.2f}s")
        logger.info(f"  • Average time per dataset: {self.total_time/len(self.datasets):.2f}s")
        
        logger.info(f"\nDetailed Results:")
        for dataset, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            logger.info(f"  {status_emoji} {dataset}:")
            logger.info(f"      Status: {result['status']}")
            logger.info(f"      Time: {result.get('execution_time', 'N/A'):.2f}s")
            
            if result['status'] == 'success' and 'metrics' in result:
                metrics = result['metrics']
                logger.info(f"      Accuracy: {metrics.get('final_accuracy', 'N/A')}")
                logger.info(f"      Loss: {metrics.get('final_loss', 'N/A')}")
            elif result['status'] != 'success':
                logger.info(f"      Error: {result.get('error', 'Unknown')}")
        
        # Save detailed results to JSON
        results_file = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'simulation_config': {
                        'num_clients': self.num_clients,
                        'num_rounds': self.num_rounds,
                        'datasets': self.datasets,
                        'total_time': self.total_time,
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': self.results
                }, f, indent=2)
            logger.info(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {str(e)}")
        
        logger.info("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-dataset federated learning simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with 5 clients, 3 rounds, using 3 datasets
  python run_multi_dataset_simulation.py --clients 5 --rounds 3 --datasets mnist iot_general custom_csv
  
  # Run simulation with default settings
  python run_multi_dataset_simulation.py
  
  # Run with custom configuration
  python run_multi_dataset_simulation.py --clients 10 --rounds 5 --datasets mnist iot_general
        """
    )
    
    parser.add_argument(
        '--clients', '-c',
        type=int,
        default=5,
        help='Number of federated learning clients (default: 5)'
    )
    
    parser.add_argument(
        '--rounds', '-r',
        type=int,
        default=3,
        help='Number of federated learning rounds (default: 3)'
    )
    
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['mnist', 'iot_general', 'custom_csv'],
        help='List of datasets to use (default: mnist iot_general custom_csv)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List datasets if requested
    if args.list_datasets:
        try:
            sim = MultiDatasetSimulation(1, 1, [])  # Dummy values for initialization
            logger.info("Available datasets:")
            for dataset in sim.available_datasets:
                logger.info(f"  • {dataset}")
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
        return
    
    # Validate arguments
    if args.clients <= 0:
        logger.error("Number of clients must be positive")
        sys.exit(1)
    
    if args.rounds <= 0:
        logger.error("Number of rounds must be positive")
        sys.exit(1)
    
    if not args.datasets:
        logger.error("At least one dataset must be specified")
        sys.exit(1)
    
    try:
        # Create and run simulation
        simulation = MultiDatasetSimulation(args.clients, args.rounds, args.datasets)
        results = simulation.run_all_experiments()
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r['status'] != 'success')
        if failed_count > 0:
            logger.warning(f"Simulation completed with {failed_count} failed experiments")
            sys.exit(1)
        else:
            logger.info("All experiments completed successfully!")
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
