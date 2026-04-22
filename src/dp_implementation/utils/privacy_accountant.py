"""Privacy accounting utilities for differential privacy."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class PrivacyAccountant:
    """Privacy accountant for tracking privacy budget consumption.
    
    This class provides methods for computing and tracking privacy loss
    in differential privacy mechanisms, including composition and amplification.
    """
    
    def __init__(self, delta: float = 1e-5) -> None:
        """Initialize privacy accountant.
        
        Args:
            delta: Privacy parameter delta (failure probability).
        """
        self.delta = delta
        self.epsilon_history: List[float] = []
        self.delta_history: List[float] = []
    
    def add_mechanism(self, epsilon: float, delta: Optional[float] = None) -> None:
        """Add a mechanism to the privacy budget.
        
        Args:
            epsilon: Privacy parameter epsilon for this mechanism.
            delta: Privacy parameter delta for this mechanism. If None, uses default.
        """
        if delta is None:
            delta = self.delta
        
        self.epsilon_history.append(epsilon)
        self.delta_history.append(delta)
    
    def get_total_privacy_loss(self) -> Tuple[float, float]:
        """Get total privacy loss from all mechanisms.
        
        Returns:
            Tuple of (total_epsilon, total_delta).
        """
        if not self.epsilon_history:
            return 0.0, 0.0
        
        # Basic composition
        total_epsilon = sum(self.epsilon_history)
        total_delta = sum(self.delta_history)
        
        return total_epsilon, total_delta
    
    def get_advanced_composition(self, k: int) -> Tuple[float, float]:
        """Get privacy loss using advanced composition theorem.
        
        Args:
            k: Number of mechanisms.
            
        Returns:
            Tuple of (total_epsilon, total_delta) using advanced composition.
        """
        if not self.epsilon_history:
            return 0.0, 0.0
        
        epsilon_list = np.array(self.epsilon_history)
        delta_list = np.array(self.delta_history)
        
        # Advanced composition theorem
        total_epsilon = np.sqrt(2 * k * np.log(1 / self.delta)) * np.max(epsilon_list) + k * np.max(epsilon_list) * (np.exp(np.max(epsilon_list)) - 1)
        total_delta = k * np.max(delta_list) + self.delta
        
        return total_epsilon, total_delta
    
    def get_moments_accountant(self, noise_multiplier: float, steps: int) -> Tuple[float, float]:
        """Get privacy loss using moments accountant (for DP-SGD).
        
        Args:
            noise_multiplier: Noise multiplier used in DP-SGD.
            steps: Number of training steps.
            
        Returns:
            Tuple of (epsilon, delta) using moments accountant.
        """
        # Simplified moments accountant approximation
        # In practice, you would use the actual moments accountant implementation
        epsilon = steps / (noise_multiplier ** 2)
        delta = self.delta
        
        return epsilon, delta
    
    def reset(self) -> None:
        """Reset privacy budget history."""
        self.epsilon_history = []
        self.delta_history = []
    
    def get_privacy_report(self) -> Dict[str, float]:
        """Get comprehensive privacy report.
        
        Returns:
            Dictionary containing privacy metrics.
        """
        basic_epsilon, basic_delta = self.get_total_privacy_loss()
        
        report = {
            "basic_composition_epsilon": basic_epsilon,
            "basic_composition_delta": basic_delta,
            "mechanism_count": len(self.epsilon_history),
            "max_epsilon": max(self.epsilon_history) if self.epsilon_history else 0.0,
            "min_epsilon": min(self.epsilon_history) if self.epsilon_history else 0.0,
            "mean_epsilon": np.mean(self.epsilon_history) if self.epsilon_history else 0.0,
        }
        
        if len(self.epsilon_history) > 1:
            adv_epsilon, adv_delta = self.get_advanced_composition(len(self.epsilon_history))
            report.update({
                "advanced_composition_epsilon": adv_epsilon,
                "advanced_composition_delta": adv_delta,
            })
        
        return report


def compute_sensitivity(data: np.ndarray, function: str = "mean") -> float:
    """Compute sensitivity for a given function.
    
    Args:
        data: Input data array.
        function: Function type ("mean", "sum", "count", "max", "min").
        
    Returns:
        Sensitivity value.
    """
    if function == "mean":
        return (np.max(data) - np.min(data)) / len(data)
    elif function == "sum":
        return np.max(data) - np.min(data)
    elif function == "count":
        return 1.0
    elif function == "max":
        return np.max(data) - np.min(data)
    elif function == "min":
        return np.max(data) - np.min(data)
    else:
        raise ValueError(f"Unknown function: {function}")


def compute_noise_scale(sensitivity: float, epsilon: float, delta: float = 0.0) -> float:
    """Compute noise scale for Gaussian mechanism.
    
    Args:
        sensitivity: Function sensitivity.
        epsilon: Privacy parameter epsilon.
        delta: Privacy parameter delta.
        
    Returns:
        Noise scale for Gaussian mechanism.
    """
    if delta == 0.0:
        # Laplace mechanism
        return sensitivity / epsilon
    else:
        # Gaussian mechanism
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def compute_laplace_noise(scale: float, size: int = 1) -> np.ndarray:
    """Generate Laplace noise.
    
    Args:
        scale: Noise scale parameter.
        size: Number of noise samples to generate.
        
    Returns:
        Array of Laplace noise samples.
    """
    return np.random.laplace(0, scale, size)


def compute_gaussian_noise(scale: float, size: int = 1) -> np.ndarray:
    """Generate Gaussian noise.
    
    Args:
        scale: Noise scale parameter.
        size: Number of noise samples to generate.
        
    Returns:
        Array of Gaussian noise samples.
    """
    return np.random.normal(0, scale, size)
