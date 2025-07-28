"""Simple configuration utilities without external dependencies."""

import os
import json
from typing import Any, Dict, Optional
from pathlib import Path


class SimpleConfig:
    """Simple configuration class without external dependencies."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SimpleConfig':
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls(config_dict)
        except Exception:
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()


def get_env_var(var_name: str, default: Any = None) -> Any:
    """Get environment variable with default."""
    return os.getenv(var_name, default)


def load_config(config_path: Optional[str] = None) -> SimpleConfig:
    """Load configuration from file or create empty config."""
    if config_path and Path(config_path).exists():
        return SimpleConfig.from_file(config_path)
    return SimpleConfig()


# For backward compatibility
Config = SimpleConfig
