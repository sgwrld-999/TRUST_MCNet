#!/usr/bin/env python3
"""
Example client for TRUST_MCNet Enhanced API Server

This script demonstrates how to interact with the enhanced API server 
to manage dynamic trust thresholds and monitor trust metrics.
"""

import argparse
import logging
import os
import sys
import json
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TRUST_MCNet API Client Example")
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="API server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="API server port"
    )
    
    parser.add_argument(
        "--action",
        type=str,
        choices=["get-threshold", "update-threshold", "get-stats", "analyze-threshold"],
        default="get-threshold",
        help="Action to perform"
    )
    
    return parser.parse_args()


def get_threshold(base_url):
    """Get current threshold configuration."""
    response = requests.get(f"{base_url}/threshold")
    if response.status_code == 200:
        data = response.json()
        logger.info("Current Threshold Configuration:")
        logger.info(f"  - Current threshold: {data['current_threshold']}")
        logger.info(f"  - Target accuracy: {data['target_accuracy']}")
        logger.info(f"  - Adaptation enabled: {data['adaptation_enabled']}")
        logger.info(f"  - Dynamic threshold enabled: {data['dynamic_threshold_enabled']}")
        
        if data['dynamic_threshold_enabled'] and data['dynamic_config']:
            config = data['dynamic_config']
            logger.info("Dynamic Threshold Configuration:")
            logger.info(f"  - Target trusted ratio: {config['target_trusted_ratio']}")
            logger.info(f"  - Min trusted clients: {config['min_trusted_clients']}")
            logger.info(f"  - Threshold range: [{config['min_threshold']}, {config['max_threshold']}]")
            logger.info("Weights:")
            logger.info(f"  - Percentile: {config['percentile_weight']}")
            logger.info(f"  - Statistical: {config['statistical_weight']}")
            logger.info(f"  - Adaptive: {config['adaptive_weight']}")
        
        return data
    else:
        logger.error(f"Failed to get threshold: {response.text}")
        return None


def update_dynamic_threshold(base_url):
    """Update dynamic threshold configuration."""
    config = {
        "config": {
            "target_trusted_ratio": 0.7,
            "min_trusted_clients": 3,
            "min_threshold": 0.15,
            "max_threshold": 0.85,
            "percentile_weight": 0.3,
            "statistical_weight": 0.5,
            "adaptive_weight": 0.2
        },
        "enable_dynamic_threshold": True,
        "reason": "Example client: Optimizing for better client selection"
    }
    
    logger.info("Updating dynamic threshold configuration...")
    response = requests.post(
        f"{base_url}/threshold/dynamic",
        headers={"Content-Type": "application/json"},
        json=config
    )
    
    if response.status_code == 200:
        logger.info("Dynamic threshold configuration updated successfully")
        return response.json()
    else:
        logger.error(f"Failed to update dynamic threshold: {response.text}")
        return None


def get_trust_stats(base_url):
    """Get trust statistics."""
    response = requests.get(f"{base_url}/trust/stats")
    if response.status_code == 200:
        data = response.json()
        logger.info("Trust Statistics:")
        logger.info(f"  - Total clients: {data['total_clients']}")
        logger.info(f"  - Mean trust: {data['mean_trust']}")
        logger.info(f"  - Trust range: [{data['min_trust']}, {data['max_trust']}]")
        logger.info(f"  - Clients quarantined: {data['clients_quarantined']}")
        logger.info(f"  - Latest round: {data['latest_round']}")
        
        if 'trust_score_distribution' in data and data['trust_score_distribution']:
            logger.info("Trust Score Distribution:")
            for bin_range, count in data['trust_score_distribution'].items():
                logger.info(f"  - {bin_range}: {count} clients")
        
        return data
    else:
        logger.error(f"Failed to get trust statistics: {response.text}")
        return None


def analyze_threshold(base_url):
    """Analyze threshold impact."""
    response = requests.get(f"{base_url}/analysis/threshold?rounds=10")
    if response.status_code == 200:
        data = response.json()
        logger.info("Threshold Impact Analysis:")
        logger.info(f"Analysis summary: {data['analysis_summary']}")
        
        if 'threshold_performance_correlation' in data and data['threshold_performance_correlation'] is not None:
            corr = data['threshold_performance_correlation']
            logger.info(f"Correlation between threshold and performance: {corr:.3f}")
            if corr > 0.5:
                logger.info("Strong positive correlation - higher thresholds likely improve performance")
            elif corr < -0.5:
                logger.info("Strong negative correlation - consider lowering thresholds")
            else:
                logger.info("No strong correlation - threshold changes have limited impact")
        
        return data
    else:
        logger.error(f"Failed to analyze threshold impact: {response.text}")
        return None


def main():
    """Main entry point."""
    args = parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    logger.info(f"Connecting to API server at {base_url}")
    
    if args.action == "get-threshold":
        get_threshold(base_url)
    elif args.action == "update-threshold":
        update_dynamic_threshold(base_url)
    elif args.action == "get-stats":
        get_trust_stats(base_url)
    elif args.action == "analyze-threshold":
        analyze_threshold(base_url)


if __name__ == "__main__":
    main()
