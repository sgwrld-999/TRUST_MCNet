"""
Real-time Trust Monitoring Dashboard for TRUST_MCNet

This module provides comprehensive trust monitoring capabilities including
metrics tracking, visualization generation, and real-time dashboard updates.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns

logger = logging.getLogger(__name__)


class TrustMetricsTracker:
    """
    Tracks and manages trust-related metrics over time.
    
    Provides comprehensive tracking of trust scores, performance metrics,
    and adaptation events with efficient storage and retrieval.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of historical entries to maintain
        """
        self.max_history = max_history
        
        # Time-series data storage
        self.trust_history = deque(maxlen=max_history)
        self.performance_history = deque(maxlen=max_history)
        self.adaptation_history = deque(maxlen=max_history)
        
        # Client-specific tracking
        self.client_trust_scores = defaultdict(lambda: deque(maxlen=100))
        self.client_performance = defaultdict(lambda: deque(maxlen=100))
        
        # Aggregated statistics
        self.round_stats = {}
        self.global_stats = {
            'total_rounds': 0,
            'total_adaptations': 0,
            'avg_trust_score': 0.0,
            'trust_score_std': 0.0
        }
        
        logger.info(f"Initialized TrustMetricsTracker with max_history={max_history}")
    
    def log_round_metrics(
        self, 
        round_num: int, 
        trust_scores: List[float],
        performance_metrics: Dict[str, Any],
        adaptation_event: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log metrics for a completed round.
        
        Args:
            round_num: Round number
            trust_scores: List of client trust scores
            performance_metrics: Performance metrics for the round
            adaptation_event: Optional adaptation event information
        """
        timestamp = datetime.now()
        
        # Log trust distribution
        if trust_scores:
            trust_entry = {
                'timestamp': timestamp,
                'round': round_num,
                'scores': trust_scores.copy(),
                'mean': np.mean(trust_scores),
                'std': np.std(trust_scores),
                'min': np.min(trust_scores),
                'max': np.max(trust_scores),
                'median': np.median(trust_scores)
            }
            self.trust_history.append(trust_entry)
        
        # Log performance metrics
        performance_entry = {
            'timestamp': timestamp,
            'round': round_num,
            **performance_metrics
        }
        self.performance_history.append(performance_entry)
        
        # Log adaptation event if provided
        if adaptation_event:
            adaptation_entry = {
                'timestamp': timestamp,
                'round': round_num,
                **adaptation_event
            }
            self.adaptation_history.append(adaptation_entry)
            self.global_stats['total_adaptations'] += 1
        
        # Update global statistics
        self.global_stats['total_rounds'] = round_num
        if trust_scores:
            self.global_stats['avg_trust_score'] = np.mean([
                entry['mean'] for entry in self.trust_history
            ])
            self.global_stats['trust_score_std'] = np.std([
                entry['mean'] for entry in self.trust_history
            ])
        
        logger.debug(f"Logged metrics for round {round_num}")
    
    def log_client_metrics(
        self, 
        client_id: str, 
        trust_score: float, 
        performance_metrics: Dict[str, Any]
    ) -> None:
        """
        Log client-specific metrics.
        
        Args:
            client_id: Client identifier
            trust_score: Trust score for the client
            performance_metrics: Client performance metrics
        """
        timestamp = datetime.now()
        
        # Store client trust score
        self.client_trust_scores[client_id].append({
            'timestamp': timestamp,
            'trust_score': trust_score
        })
        
        # Store client performance
        self.client_performance[client_id].append({
            'timestamp': timestamp,
            **performance_metrics
        })
    
    def log_trust_distribution(self, round_metrics: Dict[str, Any]) -> None:
        """
        Log trust score distribution for a round.
        
        Args:
            round_metrics: Round metrics containing trust information
        """
        trust_scores = round_metrics.get('trust_scores', [])
        round_num = round_metrics.get('round', 0)
        
        if trust_scores:
            self.log_round_metrics(
                round_num=round_num,
                trust_scores=trust_scores,
                performance_metrics=round_metrics
            )
    
    def get_trust_statistics(self, last_n_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Get trust statistics over specified period.
        
        Args:
            last_n_rounds: Number of recent rounds to analyze (None for all)
            
        Returns:
            Dictionary with trust statistics
        """
        if not self.trust_history:
            return {}
        
        # Get relevant entries
        entries = list(self.trust_history)
        if last_n_rounds:
            entries = entries[-last_n_rounds:]
        
        if not entries:
            return {}
        
        # Calculate statistics
        all_scores = []
        means = []
        stds = []
        
        for entry in entries:
            all_scores.extend(entry['scores'])
            means.append(entry['mean'])
            stds.append(entry['std'])
        
        return {
            'overall_mean': np.mean(all_scores),
            'overall_std': np.std(all_scores),
            'overall_min': np.min(all_scores),
            'overall_max': np.max(all_scores),
            'round_means': means,
            'round_stds': stds,
            'num_rounds': len(entries),
            'total_scores': len(all_scores)
        }
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to save metrics file
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'global_stats': self.global_stats,
            'trust_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.trust_history
            ],
            'performance_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.performance_history
            ],
            'adaptation_history': [
                {**entry, 'timestamp': entry['timestamp'].isoformat()}
                for entry in self.adaptation_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported metrics to {filepath}")


class TrustDashboard:
    """
    Real-time trust monitoring dashboard.
    
    Provides comprehensive visualization and monitoring capabilities for
    trust-weighted federated learning with real-time updates and insights.
    """
    
    def __init__(self, strategy, output_dir: str = "trust_dashboard"):
        """
        Initialize trust dashboard.
        
        Args:
            strategy: Federated learning strategy to monitor
            output_dir: Directory for dashboard outputs
        """
        self.strategy = strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_tracker = TrustMetricsTracker()
        
        # Dashboard state
        self.is_running = False
        self.update_thread = None
        self.update_interval = 30  # seconds
        
        # Visualization settings
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.figure_size = (12, 8)
        
        logger.info(f"Initialized TrustDashboard with output_dir={output_dir}")
    
    def start_monitoring(self, update_interval: int = 30) -> None:
        """
        Start real-time monitoring.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.update_interval = update_interval
        self.is_running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(
            target=self._update_loop, 
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"Started trust monitoring with {update_interval}s update interval")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("Stopped trust monitoring")
    
    def update_dashboard(self, round_metrics: Dict[str, Any]) -> None:
        """
        Update dashboard with new round metrics.
        
        Args:
            round_metrics: Metrics from completed round
        """
        # Log metrics
        self.metrics_tracker.log_trust_distribution(round_metrics)
        
        # Log adaptation events if present
        if hasattr(self.strategy, 'get_adaptation_status'):
            adaptation_status = self.strategy.get_adaptation_status()
            if adaptation_status.get('rounds_since_adaptation', float('inf')) == 0:
                # Recent adaptation occurred
                self.metrics_tracker.log_round_metrics(
                    round_num=round_metrics.get('round', 0),
                    trust_scores=round_metrics.get('trust_scores', []),
                    performance_metrics=round_metrics,
                    adaptation_event=adaptation_status
                )
        
        # Generate visualizations
        self.generate_trust_visualizations()
        
        # Update status files
        self._update_status_files()
        
        logger.debug(f"Updated dashboard for round {round_metrics.get('round', 'unknown')}")
    
    def generate_trust_visualizations(self) -> None:
        """Generate comprehensive trust visualizations."""
        try:
            # Generate individual plots
            self._plot_trust_evolution()
            self._plot_trust_distribution()
            self._plot_performance_vs_trust()
            self._plot_adaptation_timeline()
            
            # Generate combined dashboard
            self._generate_combined_dashboard()
            
            logger.debug("Generated trust visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_trust_evolution(self) -> None:
        """Plot trust score evolution over time."""
        if not self.metrics_tracker.trust_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                       sharex=True)
        
        # Extract data
        rounds = [entry['round'] for entry in self.metrics_tracker.trust_history]
        means = [entry['mean'] for entry in self.metrics_tracker.trust_history]
        stds = [entry['std'] for entry in self.metrics_tracker.trust_history]
        mins = [entry['min'] for entry in self.metrics_tracker.trust_history]
        maxs = [entry['max'] for entry in self.metrics_tracker.trust_history]
        
        # Plot mean trust scores
        ax1.plot(rounds, means, 'b-', linewidth=2, label='Mean Trust Score')
        ax1.fill_between(rounds, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax1.set_ylabel('Trust Score')
        ax1.set_title('Trust Score Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot trust score range
        ax2.fill_between(rounds, mins, maxs, alpha=0.4, color='green', 
                        label='Trust Score Range')
        ax2.plot(rounds, means, 'r-', linewidth=1, label='Mean')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Trust Score Range')
        ax2.set_title('Trust Score Distribution Range')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trust_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_trust_distribution(self) -> None:
        """Plot current trust score distribution."""
        if not self.metrics_tracker.trust_history:
            return
        
        # Get latest trust scores
        latest_entry = self.metrics_tracker.trust_history[-1]
        trust_scores = latest_entry['scores']
        
        if not trust_scores:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Histogram
        ax1.hist(trust_scores, bins=20, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        ax1.axvline(np.mean(trust_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(trust_scores):.3f}')
        ax1.axvline(np.median(trust_scores), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(trust_scores):.3f}')
        ax1.set_xlabel('Trust Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Trust Score Distribution (Round {latest_entry["round"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(trust_scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax2.set_ylabel('Trust Score')
        ax2.set_title('Trust Score Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trust_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_vs_trust(self) -> None:
        """Plot performance metrics vs trust scores."""
        if len(self.metrics_tracker.performance_history) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Extract data
        rounds = [entry['round'] for entry in self.metrics_tracker.performance_history]
        accuracies = [entry.get('avg_accuracy', 0) for entry in self.metrics_tracker.performance_history]
        trust_means = [entry['mean'] for entry in self.metrics_tracker.trust_history 
                      if entry['round'] in rounds]
        
        # Accuracy vs Round
        ax1.plot(rounds, accuracies, 'g-', linewidth=2, marker='o', label='Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Trust vs Accuracy correlation
        if len(trust_means) == len(accuracies):
            ax2.scatter(trust_means, accuracies, alpha=0.7, s=50)
            ax2.set_xlabel('Mean Trust Score')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Trust Score vs Performance Correlation')
            
            # Add correlation coefficient
            if len(trust_means) > 1:
                corr = np.corrcoef(trust_means, accuracies)[0, 1]
                ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_vs_trust.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_adaptation_timeline(self) -> None:
        """Plot adaptation events timeline."""
        if not hasattr(self.strategy, 'get_adaptation_status'):
            return
        
        adaptation_events = list(self.metrics_tracker.adaptation_history)
        if not adaptation_events:
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Extract adaptation data
        rounds = [event['round'] for event in adaptation_events]
        thresholds = [event.get('current_trust_threshold', 0) for event in adaptation_events]
        
        # Plot threshold changes
        ax.plot(rounds, thresholds, 'ro-', linewidth=2, markersize=8, 
               label='Trust Threshold')
        
        # Add target accuracy line if available
        target_accuracy = getattr(self.strategy, 'target_accuracy', None)
        if target_accuracy:
            ax.axhline(target_accuracy, color='green', linestyle='--', 
                      linewidth=2, label=f'Target Accuracy: {target_accuracy}')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Trust Threshold')
        ax.set_title('Adaptive Trust Threshold Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'adaptation_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_combined_dashboard(self) -> None:
        """Generate combined dashboard with all metrics."""
        if not self.metrics_tracker.trust_history:
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Trust evolution (top row, span 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        rounds = [entry['round'] for entry in self.metrics_tracker.trust_history]
        means = [entry['mean'] for entry in self.metrics_tracker.trust_history]
        ax1.plot(rounds, means, 'b-', linewidth=2)
        ax1.set_title('Trust Score Evolution')
        ax1.set_ylabel('Mean Trust Score')
        ax1.grid(True, alpha=0.3)
        
        # Current trust distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        latest_scores = self.metrics_tracker.trust_history[-1]['scores']
        ax2.hist(latest_scores, bins=15, alpha=0.7, color='skyblue')
        ax2.set_title('Current Trust Distribution')
        ax2.set_xlabel('Trust Score')
        
        # Performance metrics (middle row, span 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        if self.metrics_tracker.performance_history:
            perf_rounds = [entry['round'] for entry in self.metrics_tracker.performance_history]
            accuracies = [entry.get('avg_accuracy', 0) for entry in self.metrics_tracker.performance_history]
            ax3.plot(perf_rounds, accuracies, 'g-', linewidth=2)
            ax3.set_title('Model Performance')
            ax3.set_ylabel('Accuracy')
            ax3.grid(True, alpha=0.3)
        
        # Adaptation status (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        if hasattr(self.strategy, 'get_adaptation_status'):
            status = self.strategy.get_adaptation_status()
            ax4.text(0.1, 0.8, f"Current Threshold: {status.get('current_trust_threshold', 0):.3f}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f"Target Accuracy: {status.get('target_accuracy', 0):.3f}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f"Rounds Since Adapt: {status.get('rounds_since_adaptation', 0)}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Adaptation Status')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        # Summary statistics (bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        stats = self.metrics_tracker.get_trust_statistics()
        summary_text = f"""
        Global Statistics:
        • Total Rounds: {self.metrics_tracker.global_stats['total_rounds']}
        • Total Adaptations: {self.metrics_tracker.global_stats['total_adaptations']}
        • Overall Mean Trust: {stats.get('overall_mean', 0):.3f} ± {stats.get('overall_std', 0):.3f}
        • Trust Range: [{stats.get('overall_min', 0):.3f}, {stats.get('overall_max', 0):.3f}]
        • Active Clients: {len(self.metrics_tracker.client_trust_scores)}
        """
        ax5.text(0.05, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace')
        ax5.set_title('Summary Statistics')
        ax5.axis('off')
        
        plt.suptitle('TRUST_MCNet Real-time Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _update_loop(self) -> None:
        """Background update loop for real-time monitoring."""
        while self.is_running:
            try:
                # Generate periodic updates
                self.generate_trust_visualizations()
                self._update_status_files()
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _update_status_files(self) -> None:
        """Update status files with current metrics."""
        try:
            # Export metrics
            self.metrics_tracker.export_metrics(
                self.output_dir / 'trust_metrics.json'
            )
            
            # Create status summary
            status = {
                'timestamp': datetime.now().isoformat(),
                'dashboard_status': 'running' if self.is_running else 'stopped',
                'total_rounds': self.metrics_tracker.global_stats['total_rounds'],
                'total_adaptations': self.metrics_tracker.global_stats['total_adaptations'],
                'latest_trust_stats': self.metrics_tracker.get_trust_statistics(last_n_rounds=5)
            }
            
            if hasattr(self.strategy, 'get_adaptation_status'):
                status['adaptation_status'] = self.strategy.get_adaptation_status()
            
            with open(self.output_dir / 'status.json', 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error updating status files: {e}")
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL for web viewing."""
        dashboard_path = self.output_dir / 'dashboard.png'
        return f"file://{dashboard_path.absolute()}"
