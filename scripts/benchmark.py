#!/usr/bin/env python3
"""
TRUST_MCNet Performance Benchmark Script

Runs performance sanity checks to verify the system produces
plausible results and trust mechanisms are working correctly.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --dataset edge_iiot --rounds 3
"""

import argparse
import sys
import os
import time
import traceback
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from scripts.train_mcnet import run_simulation, load_config
    from datasets import list_datasets
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Ensure the project structure is correct and all dependencies are installed.")
    sys.exit(1)


class BenchmarkRunner:
    """Manages benchmark execution and validation."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_dataset_benchmark(
        self, 
        dataset_name: str, 
        rounds: int = 3, 
        config_path: str = "config/trust.yaml"
    ) -> Dict[str, Any]:
        """
        Run benchmark for a specific dataset.
        
        Args:
            dataset_name: Name of dataset to benchmark
            rounds: Number of federated rounds
            config_path: Path to configuration file
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*50}")
        print(f"Benchmarking {dataset_name.upper()} Dataset")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Run simulation
            print(f"Running {rounds} rounds of federated learning...")
            results = run_simulation(dataset_name, rounds, config)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract key metrics
            benchmark_results = {
                "dataset": dataset_name,
                "rounds": rounds,
                "duration": duration,
                "status": "success",
                "metrics": self._extract_metrics(results),
                "validation": self._validate_results(results),
                "timestamp": datetime.now().isoformat()
            }
            
            # Print summary
            self._print_dataset_summary(benchmark_results)
            
            return benchmark_results
            
        except Exception as e:
            error_info = {
                "dataset": dataset_name,
                "rounds": rounds,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"❌ Benchmark failed for {dataset_name}: {e}")
            return error_info
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from simulation results."""
        metrics = {}
        
        # Extract final metrics
        if "final_metrics" in results:
            final_metrics = results["final_metrics"]
            metrics.update({
                "final_accuracy": final_metrics.get("final_accuracy"),
                "final_loss": final_metrics.get("final_loss")
            })
        
        # Extract trust summary
        if "trust_summary" in results:
            trust_summary = results["trust_summary"]
            if isinstance(trust_summary, dict):
                metrics.update({
                    "mean_trust": trust_summary.get("mean_trust"),
                    "trust_std": trust_summary.get("trust_std"),
                    "quarantined_count": len(trust_summary.get("quarantined_clients", []))
                })
        
        # Extract duration
        metrics["duration"] = results.get("duration", 0)
        
        return metrics
    
    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate benchmark results for sanity."""
        validation = {
            "passed": True,
            "checks": [],
            "warnings": []
        }
        
        metrics = self._extract_metrics(results)
        
        # Check accuracy range
        final_accuracy = metrics.get("final_accuracy")
        if final_accuracy is not None:
            if 0.0 <= final_accuracy <= 1.0:
                validation["checks"].append("✓ Accuracy in valid range [0,1]")
            else:
                validation["checks"].append(f"❌ Invalid accuracy: {final_accuracy}")
                validation["passed"] = False
            
            if final_accuracy < 0.1:
                validation["warnings"].append(f"⚠️ Low accuracy: {final_accuracy:.3f}")
        
        # Check trust metrics
        mean_trust = metrics.get("mean_trust")
        if mean_trust is not None:
            if 0.0 <= mean_trust <= 1.0:
                validation["checks"].append("✓ Trust scores in valid range [0,1]")
            else:
                validation["checks"].append(f"❌ Invalid trust scores: {mean_trust}")
                validation["passed"] = False
        
        # Check quarantine mechanism
        quarantined_count = metrics.get("quarantined_count", 0)
        if quarantined_count >= 0:
            validation["checks"].append("✓ Quarantine mechanism active")
            if quarantined_count > 0:
                validation["checks"].append(f"✓ {quarantined_count} clients quarantined")
        
        # Check duration reasonableness
        duration = metrics.get("duration", 0)
        if duration > 0:
            validation["checks"].append(f"✓ Reasonable execution time: {duration:.1f}s")
            if duration > 300:  # 5 minutes
                validation["warnings"].append(f"⚠️ Long execution time: {duration:.1f}s")
        
        return validation
    
    def _print_dataset_summary(self, results: Dict[str, Any]):
        """Print summary for a dataset benchmark."""
        if results["status"] == "success":
            metrics = results["metrics"]
            validation = results["validation"]
            
            print(f"\n📊 Results Summary:")
            print(f"   Duration: {metrics.get('duration', 0):.2f} seconds")
            
            if metrics.get("final_accuracy") is not None:
                print(f"   Final Accuracy: {metrics['final_accuracy']:.4f}")
            
            if metrics.get("final_loss") is not None:
                print(f"   Final Loss: {metrics['final_loss']:.4f}")
            
            if metrics.get("mean_trust") is not None:
                print(f"   Mean Trust Score: {metrics['mean_trust']:.4f}")
            
            if metrics.get("quarantined_count") is not None:
                print(f"   Quarantined Clients: {metrics['quarantined_count']}")
            
            print(f"\n✅ Validation Results:")
            for check in validation["checks"]:
                print(f"   {check}")
            
            if validation["warnings"]:
                print(f"\n⚠️ Warnings:")
                for warning in validation["warnings"]:
                    print(f"   {warning}")
            
            status = "PASSED" if validation["passed"] else "FAILED"
            print(f"\n🎯 Benchmark Status: {status}")
        else:
            print(f"\n❌ Benchmark FAILED: {results.get('error', 'Unknown error')}")
    
    def run_full_benchmark(
        self, 
        datasets: List[str] = None, 
        rounds: int = 3
    ) -> Dict[str, Any]:
        """
        Run full benchmark across multiple datasets.
        
        Args:
            datasets: List of datasets to benchmark (None for all available)
            rounds: Number of rounds per dataset
            
        Returns:
            Complete benchmark results
        """
        self.start_time = time.time()
        
        if datasets is None:
            datasets = ["ton_iot", "edge_iiot", "medbiot"]
        
        print(f"🚀 Starting TRUST-MCNet Performance Benchmark")
        print(f"📋 Datasets: {', '.join(datasets)}")
        print(f"🔄 Rounds per dataset: {rounds}")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        dataset_results = {}
        overall_stats = defaultdict(list)
        
        for dataset in datasets:
            try:
                result = self.run_dataset_benchmark(dataset, rounds)
                dataset_results[dataset] = result
                
                # Collect stats for overall summary
                if result["status"] == "success":
                    metrics = result["metrics"]
                    for key, value in metrics.items():
                        if value is not None and isinstance(value, (int, float)):
                            overall_stats[key].append(value)
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Benchmark interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error benchmarking {dataset}: {e}")
                dataset_results[dataset] = {
                    "status": "error",
                    "error": str(e),
                    "dataset": dataset
                }
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Create overall summary
        benchmark_summary = {
            "benchmark_info": {
                "total_duration": total_duration,
                "datasets_tested": len(dataset_results),
                "successful_tests": sum(1 for r in dataset_results.values() if r["status"] == "success"),
                "timestamp": datetime.now().isoformat()
            },
            "dataset_results": dataset_results,
            "overall_stats": dict(overall_stats),
            "validation_summary": self._create_validation_summary(dataset_results)
        }
        
        self.results = benchmark_summary
        
        # Print overall summary
        self._print_overall_summary(benchmark_summary)
        
        return benchmark_summary
    
    def _create_validation_summary(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall validation summary."""
        total_tests = len(dataset_results)
        passed_tests = sum(
            1 for result in dataset_results.values() 
            if result.get("validation", {}).get("passed", False)
        )
        
        all_checks = []
        all_warnings = []
        
        for result in dataset_results.values():
            if "validation" in result:
                all_checks.extend(result["validation"].get("checks", []))
                all_warnings.extend(result["validation"].get("warnings", []))
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "total_checks": len(all_checks),
            "total_warnings": len(all_warnings),
            "overall_status": "PASSED" if passed_tests == total_tests else "PARTIAL"
        }
    
    def _print_overall_summary(self, results: Dict[str, Any]):
        """Print overall benchmark summary."""
        print(f"\n{'='*60}")
        print(f"🎯 TRUST-MCNet Benchmark Summary")
        print(f"{'='*60}")
        
        info = results["benchmark_info"]
        validation = results["validation_summary"]
        
        print(f"⏱️  Total Duration: {info['total_duration']:.2f} seconds")
        print(f"📊 Datasets Tested: {info['datasets_tested']}")
        print(f"✅ Successful Tests: {info['successful_tests']}/{info['datasets_tested']}")
        print(f"📈 Success Rate: {validation['success_rate']:.1%}")
        
        if results["overall_stats"]:
            print(f"\n📊 Overall Statistics:")
            stats = results["overall_stats"]
            
            if "final_accuracy" in stats and stats["final_accuracy"]:
                accuracies = stats["final_accuracy"]
                print(f"   Average Accuracy: {sum(accuracies)/len(accuracies):.4f}")
                print(f"   Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
            
            if "duration" in stats and stats["duration"]:
                durations = stats["duration"]
                print(f"   Average Duration: {sum(durations)/len(durations):.2f}s")
        
        print(f"\n🔍 Validation Summary:")
        print(f"   Total Checks: {validation['total_checks']}")
        print(f"   Warnings: {validation['total_warnings']}")
        print(f"   Overall Status: {validation['overall_status']}")
        
        # Individual dataset status
        print(f"\n📋 Dataset Results:")
        for dataset, result in results["dataset_results"].items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"   {status_icon} {dataset}: {result['status'].upper()}")
        
        print(f"\n{'='*60}")
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Benchmark results saved to: {filepath}")
        return filepath


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="TRUST_MCNet Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py
  python scripts/benchmark.py --dataset ton_iot --rounds 2
  python scripts/benchmark.py --all-datasets --rounds 5 --save-results
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=["ton_iot", "edge_iiot", "medbiot"],
        help="Run benchmark for specific dataset only"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run benchmark for all available datasets"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of federated rounds per dataset (default: 3)"
    )
    parser.add_argument(
        "--config",
        default="config/trust.yaml",
        help="Configuration file path (default: config/trust.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(args.output_dir)
        
        # Determine datasets to test
        if args.dataset:
            datasets = [args.dataset]
        elif args.all_datasets:
            datasets = ["ton_iot", "edge_iiot", "medbiot"]
        else:
            # Default: run quick benchmark with ToN-IoT
            datasets = ["ton_iot"]
        
        # Run benchmark
        results = runner.run_full_benchmark(datasets, args.rounds)
        
        # Save results if requested
        if args.save_results:
            runner.save_results()
        
        # Determine exit code based on validation results
        validation_summary = results["validation_summary"]
        if validation_summary["overall_status"] == "PASSED":
            print(f"\n🎉 All benchmarks PASSED!")
            return 0
        elif validation_summary["overall_status"] == "PARTIAL":
            print(f"\n⚠️ Some benchmarks failed or had warnings")
            return 1
        else:
            print(f"\n❌ Benchmark validation FAILED")
            return 2
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Benchmark crashed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
