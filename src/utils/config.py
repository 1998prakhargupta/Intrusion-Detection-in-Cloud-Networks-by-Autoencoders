"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration schema."""
    
    name: str = "NetworkIntrusionAutoencoder"
    version: str = "1.0.0"
    input_size: int = 4
    hidden_size: int = 2
    activation: str = "relu"
    dropout_rate: float = 0.1
    batch_norm: bool = False
    
    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training configuration schema."""
    
    epochs: int = Field(default=200, ge=1, le=1000)
    batch_size: int = Field(default=64, ge=1, le=1024)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    shuffle: bool = True
    seed: int = 42
    
    # Early stopping
    early_stopping_enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-6
    
    class Config:
        extra = "forbid"


class DataConfig(BaseModel):
    """Data configuration schema."""
    
    path: str
    format: str = "csv"
    encoding: str = "utf-8"
    
    # Feature configuration
    selected_features: List[str] = [
        "Duration", "Orig_bytes", "Resp_bytes", "Orig_pkts"
    ]
    target_column: str = "class"
    normal_class: str = "normal"
    
    # Preprocessing
    scaling_method: str = "minmax"
    feature_range: Tuple[float, float] = (0, 1)
    missing_strategy: str = "median"
    missing_threshold: float = 0.05
    
    class Config:
        extra = "forbid"


class ThresholdConfig(BaseModel):
    """Threshold configuration schema."""
    
    methods: List[str] = ["percentile", "statistical", "roc_optimal"]
    percentile_value: float = Field(default=95, ge=0, le=100)
    statistical_n_std: float = Field(default=2.0, gt=0)
    roc_optimization_metric: str = "f1"
    
    class Config:
        extra = "forbid"


class APIConfig(BaseModel):
    """API configuration schema."""
    
    title: str = "Network Intrusion Detection API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    
    # Security
    authentication_enabled: bool = False
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100
    
    class Config:
        extra = "forbid"


class Config:
    """Main configuration manager."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, loads default config.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_data: Dict[str, Any] = {}
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
        else:
            self.load_defaults()
            
        # Initialize config objects
        self._init_config_objects()
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f)
            
    def load_defaults(self) -> None:
        """Load default configuration."""
        self._config_data = {
            'model': {
                'name': 'NetworkIntrusionAutoencoder',
                'version': '1.0.0',
                'architecture': {
                    'input_size': 4,
                    'hidden_size': 2,
                    'activation': 'relu',
                    'dropout_rate': 0.1,
                    'batch_norm': False
                }
            },
            'training': {
                'epochs': 200,
                'batch_size': 64,
                'learning_rate': 0.01,
                'weight_decay': 1e-5,
                'validation_split': 0.2,
                'shuffle': True,
                'seed': 42,
                'early_stopping': {
                    'enabled': True,
                    'patience': 20,
                    'min_delta': 1e-6
                }
            },
            'data': {
                'features': {
                    'selected': ['Duration', 'Orig_bytes', 'Resp_bytes', 'Orig_pkts']
                },
                'target_column': 'class',
                'normal_class': 'normal',
                'preprocessing': {
                    'scaling': {
                        'method': 'minmax',
                        'feature_range': [0, 1]
                    }
                }
            },
            'thresholds': {
                'methods': ['percentile', 'statistical', 'roc_optimal'],
                'percentile': {'value': 95},
                'statistical': {'n_std': 2.0}
            }
        }
    
    def _init_config_objects(self) -> None:
        """Initialize typed configuration objects."""
        # Model config
        model_data = self._config_data.get('model', {})
        arch_data = model_data.get('architecture', {})
        
        self.model = ModelConfig(
            name=model_data.get('name', 'NetworkIntrusionAutoencoder'),
            version=model_data.get('version', '1.0.0'),
            input_size=arch_data.get('input_size', 4),
            hidden_size=arch_data.get('hidden_size', 2),
            activation=arch_data.get('activation', 'relu'),
            dropout_rate=arch_data.get('dropout_rate', 0.1),
            batch_norm=arch_data.get('batch_norm', False)
        )
        
        # Training config
        training_data = self._config_data.get('training', {})
        early_stopping = training_data.get('early_stopping', {})
        
        self.training = TrainingConfig(
            epochs=training_data.get('epochs', 200),
            batch_size=training_data.get('batch_size', 64),
            learning_rate=training_data.get('learning_rate', 0.01),
            weight_decay=training_data.get('weight_decay', 1e-5),
            validation_split=training_data.get('validation_split', 0.2),
            shuffle=training_data.get('shuffle', True),
            seed=training_data.get('seed', 42),
            early_stopping_enabled=early_stopping.get('enabled', True),
            patience=early_stopping.get('patience', 20),
            min_delta=early_stopping.get('min_delta', 1e-6)
        )
        
        # Data config (simplified for now)
        data_section = self._config_data.get('data', {})
        features_section = data_section.get('features', {})
        
        self.data = DataConfig(
            path=data_section.get('source', {}).get('path', ''),
            selected_features=features_section.get('selected', [
                'Duration', 'Orig_bytes', 'Resp_bytes', 'Orig_pkts'
            ]),
            target_column=features_section.get('target_column', 'class'),
            normal_class=features_section.get('normal_class', 'normal')
        )
        
        # Threshold config
        threshold_data = self._config_data.get('thresholds', {})
        
        self.thresholds = ThresholdConfig(
            methods=threshold_data.get('methods', ['percentile']),
            percentile_value=threshold_data.get('percentile', {}).get('value', 95),
            statistical_n_std=threshold_data.get('statistical', {}).get('n_std', 2.0),
            roc_optimization_metric=threshold_data.get('roc_optimal', {}).get(
                'optimization_metric', 'f1'
            )
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        keys = key.split('.')
        target = self._config_data
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
            
        target[keys[-1]] = value
        
        # Reinitialize config objects
        self._init_config_objects()
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Path to save configuration.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return self._config_data.copy()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or defaults.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded configuration.
    """
    return Config(config_path)


def get_env_var(name: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with optional default and validation.
    
    Args:
        name: Environment variable name.
        default: Default value if not found.
        required: Whether the variable is required.
        
    Returns:
        Environment variable value.
        
    Raises:
        ValueError: If required variable is not found.
    """
    value = os.getenv(name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{name}' not found")
        
    return value
