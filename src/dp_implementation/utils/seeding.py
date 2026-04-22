"""Deterministic seeding utilities for reproducible experiments."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    This function ensures reproducible results across different runs by setting
    seeds for Python's random module, NumPy, PyTorch, and environment variables.
    
    Args:
        seed: Random seed value to use for all generators.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PyTorch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Get the best available device for computation.
    
    Returns:
        torch.device: CUDA if available, otherwise MPS if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_timezone() -> None:
    """Set timezone for consistent log timestamps."""
    os.environ['TZ'] = 'UTC'
