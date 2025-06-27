"""
Attack simulation utilities for TRUST-MCNet federated learning.

This module implements various adversarial attack scenarios to test
the robustness of the federated learning system:
- Label flipping attacks
- Gaussian noise injection
- Model poisoning attacks
- Byzantine behavior simulation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
from enum import Enum


class AttackType(Enum):
    """Enumeration of supported attack types."""
    LABEL_FLIPPING = "label_flipping"
    GAUSSIAN_NOISE = "gaussian_noise"
    MODEL_POISONING = "model_poisoning"
    DATA_POISONING = "data_poisoning"
    BACKDOOR = "backdoor"
    FREE_RIDING = "free_riding"


class AttackSimulator:
    """
    Simulates various adversarial attacks for testing federated learning robustness.
    
    This class provides methods to simulate different types of malicious client
    behaviors to evaluate the effectiveness of trust mechanisms.
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        Initialize attack simulator.
        
        Args:
            attack_config: Configuration dictionary containing attack parameters
        """
        self.attack_config = attack_config
        self.logger = logging.getLogger(__name__)
        self.malicious_clients = set()
        
    def designate_malicious_clients(self, client_ids: List[str], 
                                  malicious_ratio: float = 0.2) -> List[str]:
        """
        Randomly designate a subset of clients as malicious.
        
        Args:
            client_ids: List of all client IDs
            malicious_ratio: Fraction of clients to make malicious
            
        Returns:
            List of malicious client IDs
        """
        num_malicious = max(1, int(len(client_ids) * malicious_ratio))
        malicious_clients = random.sample(client_ids, num_malicious)
        self.malicious_clients.update(malicious_clients)
        
        self.logger.info(f"Designated {len(malicious_clients)} malicious clients: {malicious_clients}")
        return malicious_clients
    
    def apply_attack(self, client_id: str, attack_type: AttackType,
                    data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    model_update: Optional[Dict[str, torch.Tensor]] = None,
                    **kwargs) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], 
                                     Optional[Dict[str, torch.Tensor]]]:
        """
        Apply specified attack to client data or model update.
        
        Args:
            client_id: ID of the client
            attack_type: Type of attack to apply
            data: Client's training data (features, labels)
            model_update: Client's model parameter updates
            **kwargs: Additional attack parameters
            
        Returns:
            Tuple of (modified_data, modified_model_update)
        """
        if client_id not in self.malicious_clients:
            return data, model_update
        
        modified_data = data
        modified_model_update = model_update
        
        if attack_type == AttackType.LABEL_FLIPPING and data is not None:
            modified_data = self._label_flipping_attack(data, **kwargs)
            
        elif attack_type == AttackType.GAUSSIAN_NOISE and model_update is not None:
            modified_model_update = self._gaussian_noise_attack(model_update, **kwargs)
            
        elif attack_type == AttackType.MODEL_POISONING and model_update is not None:
            modified_model_update = self._model_poisoning_attack(model_update, **kwargs)
            
        elif attack_type == AttackType.DATA_POISONING and data is not None:
            modified_data = self._data_poisoning_attack(data, **kwargs)
            
        elif attack_type == AttackType.BACKDOOR and data is not None:
            modified_data = self._backdoor_attack(data, **kwargs)
            
        elif attack_type == AttackType.FREE_RIDING and model_update is not None:
            modified_model_update = self._free_riding_attack(model_update, **kwargs)
        
        self.logger.debug(f"Applied {attack_type.value} attack to client {client_id}")
        return modified_data, modified_model_update
    
    def _label_flipping_attack(self, data: Tuple[torch.Tensor, torch.Tensor],
                              flip_ratio: float = 0.1,
                              targeted: bool = False,
                              target_class: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply label flipping attack to training data.
        
        Args:
            data: Tuple of (features, labels)
            flip_ratio: Fraction of labels to flip
            targeted: Whether to target specific class
            target_class: Target class for targeted attack
            
        Returns:
            Modified data with flipped labels
        """
        features, labels = data
        modified_labels = labels.clone()
        
        num_samples = len(labels)
        num_flip = int(num_samples * flip_ratio)
        
        if targeted:
            # Target specific class
            target_indices = (labels == target_class).nonzero(as_tuple=True)[0]
            if len(target_indices) > 0:
                flip_indices = target_indices[torch.randperm(len(target_indices))[:num_flip]]
                # Flip to opposite class (assuming binary classification)
                modified_labels[flip_indices] = 1 - modified_labels[flip_indices]
        else:
            # Random label flipping
            flip_indices = torch.randperm(num_samples)[:num_flip]
            unique_classes = torch.unique(labels)
            
            for idx in flip_indices:
                current_label = modified_labels[idx]
                # Flip to random different class
                other_classes = unique_classes[unique_classes != current_label]
                if len(other_classes) > 0:
                    modified_labels[idx] = other_classes[torch.randint(len(other_classes), (1,))]
        
        self.logger.debug(f"Label flipping: flipped {num_flip} labels")
        return features, modified_labels
    
    def _gaussian_noise_attack(self, model_update: Dict[str, torch.Tensor],
                              noise_scale: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to model parameters.
        
        Args:
            model_update: Original model parameter updates
            noise_scale: Scale of Gaussian noise
            
        Returns:
            Model update with added noise
        """
        noisy_update = {}
        
        for param_name, param_tensor in model_update.items():
            noise = torch.randn_like(param_tensor) * noise_scale
            noisy_update[param_name] = param_tensor + noise
        
        self.logger.debug(f"Gaussian noise: added noise with scale {noise_scale}")
        return noisy_update
    
    def _model_poisoning_attack(self, model_update: Dict[str, torch.Tensor],
                               poison_scale: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        Apply model poisoning by scaling up parameter updates.
        
        Args:
            model_update: Original model parameter updates
            poison_scale: Scaling factor for poisoning
            
        Returns:
            Poisoned model update
        """
        poisoned_update = {}
        
        for param_name, param_tensor in model_update.items():
            # Scale up the update to disrupt aggregation
            poisoned_update[param_name] = param_tensor * poison_scale
        
        self.logger.debug(f"Model poisoning: scaled updates by factor {poison_scale}")
        return poisoned_update
    
    def _data_poisoning_attack(self, data: Tuple[torch.Tensor, torch.Tensor],
                              poison_ratio: float = 0.1,
                              noise_scale: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data poisoning by corrupting feature values.
        
        Args:
            data: Tuple of (features, labels)
            poison_ratio: Fraction of samples to poison
            noise_scale: Scale of noise to add to features
            
        Returns:
            Data with poisoned features
        """
        features, labels = data
        modified_features = features.clone()
        
        num_samples = len(features)
        num_poison = int(num_samples * poison_ratio)
        poison_indices = torch.randperm(num_samples)[:num_poison]
        
        # Add noise to selected samples
        noise = torch.randn_like(modified_features[poison_indices]) * noise_scale
        modified_features[poison_indices] += noise
        
        self.logger.debug(f"Data poisoning: poisoned {num_poison} samples")
        return modified_features, labels
    
    def _backdoor_attack(self, data: Tuple[torch.Tensor, torch.Tensor],
                        trigger_pattern: Optional[torch.Tensor] = None,
                        target_label: int = 1,
                        poison_ratio: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply backdoor attack by inserting trigger patterns.
        
        Args:
            data: Tuple of (features, labels)
            trigger_pattern: Trigger pattern to insert
            target_label: Target label for backdoored samples
            poison_ratio: Fraction of samples to backdoor
            
        Returns:
            Data with backdoor triggers
        """
        features, labels = data
        modified_features = features.clone()
        modified_labels = labels.clone()
        
        num_samples = len(features)
        num_backdoor = int(num_samples * poison_ratio)
        backdoor_indices = torch.randperm(num_samples)[:num_backdoor]
        
        # Create simple trigger pattern if none provided
        if trigger_pattern is None:
            # Simple pattern: set last few features to specific values
            trigger_size = min(5, features.shape[-1])
            trigger_pattern = torch.ones(trigger_size)
        
        # Insert trigger pattern
        for idx in backdoor_indices:
            if len(features.shape) == 2:  # Flat features
                trigger_size = min(len(trigger_pattern), features.shape[1])
                modified_features[idx, -trigger_size:] = trigger_pattern[:trigger_size]
            else:  # Multi-dimensional features
                # Insert trigger in corner (simplified)
                trigger_size = min(len(trigger_pattern), features.shape[-1])
                modified_features[idx, ..., -trigger_size:] = trigger_pattern[:trigger_size]
            
            # Set target label
            modified_labels[idx] = target_label
        
        self.logger.debug(f"Backdoor attack: inserted triggers in {num_backdoor} samples")
        return modified_features, modified_labels
    
    def _free_riding_attack(self, model_update: Dict[str, torch.Tensor],
                           noise_scale: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        Apply free-riding attack by sending minimal updates.
        
        Args:
            model_update: Original model parameter updates
            noise_scale: Scale of minimal noise to add
            
        Returns:
            Minimal model update (free-riding)
        """
        free_riding_update = {}
        
        for param_name, param_tensor in model_update.items():
            # Send very small random updates instead of actual training updates
            small_noise = torch.randn_like(param_tensor) * noise_scale
            free_riding_update[param_name] = small_noise
        
        self.logger.debug("Free-riding attack: sent minimal updates")
        return free_riding_update
    
    def simulate_byzantine_behavior(self, model_update: Dict[str, torch.Tensor],
                                  behavior_type: str = "random") -> Dict[str, torch.Tensor]:
        """
        Simulate various Byzantine behaviors.
        
        Args:
            model_update: Original model parameter updates
            behavior_type: Type of Byzantine behavior ('random', 'opposite', 'zero')
            
        Returns:
            Byzantine model update
        """
        byzantine_update = {}
        
        for param_name, param_tensor in model_update.items():
            if behavior_type == "random":
                # Send random parameters
                byzantine_update[param_name] = torch.randn_like(param_tensor)
            elif behavior_type == "opposite":
                # Send opposite of actual update
                byzantine_update[param_name] = -param_tensor
            elif behavior_type == "zero":
                # Send zero updates
                byzantine_update[param_name] = torch.zeros_like(param_tensor)
            else:
                # Default to original update
                byzantine_update[param_name] = param_tensor
        
        self.logger.debug(f"Byzantine behavior: applied {behavior_type} behavior")
        return byzantine_update
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about applied attacks.
        
        Returns:
            Dictionary containing attack statistics
        """
        return {
            'num_malicious_clients': len(self.malicious_clients),
            'malicious_clients': list(self.malicious_clients),
            'attack_config': self.attack_config
        }
    
    def reset_malicious_clients(self):
        """Reset the set of malicious clients."""
        self.malicious_clients.clear()
        self.logger.info("Reset malicious clients list")
