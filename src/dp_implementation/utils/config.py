"""Configuration management for differential privacy implementation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the differential privacy implementation.
    
    Handles loading and validation of configuration files with support for
    YAML and OmegaConf formats.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self._config: Optional[DictConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or create default configuration."""
        if self.config_path and Path(self.config_path).exists():
            self._config = OmegaConf.load(self.config_path)
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        default_config = {
            "data": {
                "synthetic": {
                    "n_samples": 1000,
                    "n_features": 10,
                    "noise_level": 0.1,
                    "random_seed": 42
                },
                "splits": {
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15
                }
            },
            "privacy": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "mechanisms": ["laplace", "gaussian", "exponential"]
            },
            "model": {
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 100,
                "dp_sgd": {
                    "noise_multiplier": 1.1,
                    "max_grad_norm": 1.0
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "privacy_loss", "utility_loss"],
                "n_runs": 10
            },
            "visualization": {
                "figsize": [10, 6],
                "dpi": 300,
                "style": "seaborn-v0_8"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        if self._config is None:
            return default
        
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        if self._config is None:
            self._config = OmegaConf.create({})
        
        OmegaConf.set(self._config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        if self._config is None:
            return {}
        return OmegaConf.to_container(self._config, resolve=True)
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration file.
        """
        if self._config is None:
            return
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply.
        """
        if self._config is None:
            self._config = OmegaConf.create({})
        
        for key, value in updates.items():
            self.set(key, value)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying configuration object.
        
        Returns:
            OmegaConf DictConfig object.
        """
        if self._config is None:
            self._config = self._get_default_config()
        return self._config
