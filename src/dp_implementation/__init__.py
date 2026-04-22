"""Differential Privacy Implementation Package.

A comprehensive differential privacy implementation for research and education.
This package provides various DP mechanisms, privacy accounting, and evaluation tools.
"""

__version__ = "1.0.0"
__author__ = "Security & Privacy Research"

from .data import DataGenerator, DataProcessor
from .features import FeatureExtractor
from .models import DPMechanisms, DPSGD, PATE
from .defenses import PrivacyDefense
from .eval import PrivacyEvaluator, UtilityEvaluator
from .viz import PrivacyVisualizer
from .utils import PrivacyAccountant, Config

__all__ = [
    "DataGenerator",
    "DataProcessor", 
    "FeatureExtractor",
    "DPMechanisms",
    "DPSGD",
    "PATE",
    "PrivacyDefense",
    "PrivacyEvaluator",
    "UtilityEvaluator",
    "PrivacyVisualizer",
    "PrivacyAccountant",
    "Config",
]
