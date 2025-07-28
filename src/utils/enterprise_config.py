"""
Enterprise Configuration Management System for NIDS Autoencoder.

This module provides a comprehensive configuration management system that:
- Centralizes all configuration using YAML/JSON files
- Validates all settings including model hyperparameters, paths, and thresholds
- Supports environment-specific configurations (dev, staging, production)
- Provides configuration inheritance and overrides
- Ensures type safety and validation
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

# Third-party imports
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None

# Project imports
from .logger import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigFormat(Enum):
    """Configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


@dataclass
class ModelArchitectureConfig:
    """Model architecture configuration."""
    input_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16, 32, 64])
    output_dim: Optional[int] = None  # Auto-calculated if None
    activation: str = "relu"
    dropout_rate: float = 0.1
    batch_norm: bool = True
    layer_norm: bool = False
    
    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.input_dim


@dataclass  
class TrainingConfig:
    """Training configuration."""
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    early_stopping_monitor: str = "val_loss"
    
    # Validation
    validation_split: float = 0.2
    validation_freq: int = 1
    shuffle: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_freq: int = 10
    save_best_only: bool = True


@dataclass
class DataConfig:
    """Data configuration matching YAML structure."""
    # Source configuration - nested structure
    source: Dict[str, Any] = field(default_factory=lambda: {
        "path": "data/raw/CIDDS-001-external-week3_1.csv",
        "format": "csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "header": 0
    })
    
    # Features configuration - nested structure  
    features: Dict[str, Any] = field(default_factory=lambda: {
        "columns": [
            "Duration", "Orig_bytes", "Resp_bytes", "Orig_pkts",
            "Resp_pkts", "Orig_ip_bytes", "Resp_ip_bytes"
        ],
        "target_column": "class",
        "normal_class": "normal",
        "anomaly_classes": ["dos", "probe", "r2l", "u2r"],
        "categorical_columns": [],
        "datetime_columns": []
    })
    
    # Preprocessing configuration - nested structure
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "missing_values": {
            "strategy": "median",
            "threshold": 0.05
        },
        "scaling": {
            "method": "standard",
            "feature_range": [0, 1]
        },
        "outliers": {
            "detection_method": "iqr",
            "threshold": 3.0,
            "action": "clip"
        },
        "feature_engineering": {
            "enabled": True,
            "methods": ["polynomial", "interaction", "log_transform"]
        }
    })
    
    # Splitting configuration - nested structure
    splitting: Dict[str, Any] = field(default_factory=lambda: {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "stratify": True,
        "random_state": 42,
        "shuffle": True
    })
    
    # Validation configuration
    validation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "check_missing": True,
        "check_duplicates": True,
        "check_data_types": True,
        "min_samples": 1000
    })
    
    # Caching configuration
    caching: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "cache_dir": "data/cache",
        "ttl_hours": 24,
        "compression": "gzip"
    })
    
    def __post_init__(self):
        # Validate ratios sum to 1
        train_ratio = self.splitting["train_ratio"]
        val_ratio = self.splitting["val_ratio"] 
        test_ratio = self.splitting["test_ratio"]
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total}")


@dataclass
class ThresholdConfig:
    """Threshold calculation configuration."""
    methods: List[str] = field(default_factory=lambda: [
        "percentile", "statistical", "roc_optimal", "precision_recall"
    ])
    
    # Percentile method
    percentile_value: float = 95.0
    
    # Statistical method
    statistical_factor: float = 2.0
    statistical_method: str = "std"  # std, mad, iqr
    
    # ROC optimal
    roc_metric: str = "youden"  # youden, f1, balanced_accuracy
    
    # Precision-Recall
    target_precision: float = 0.95
    target_recall: float = 0.8
    
    # Default method
    default_method: str = "percentile"


@dataclass
class APIConfig:
    """API configuration."""
    # Server
    title: str = "NIDS Autoencoder API"
    description: str = "Network Intrusion Detection System using Autoencoders"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Performance
    max_batch_size: int = 1000
    timeout_seconds: int = 30
    keepalive_timeout: int = 5
    
    # Security
    authentication_enabled: bool = False
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100
    burst_size: int = 10
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    request_logging: bool = True
    
    # Documentation
    docs_enabled: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    enabled: bool = False
    
    # Connection
    host: str = "localhost"
    port: int = 5432
    database: str = "nids"
    username: str = "nids_user"
    password: str = ""  # Will be loaded from env
    
    # Pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Query settings
    echo: bool = False
    echo_pool: bool = False


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    
    # Redis
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Memory cache
    memory_enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 3600
    
    # File cache
    file_enabled: bool = True
    cache_dir: str = "cache"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Handlers
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = False
    syslog_enabled: bool = False
    
    # File settings
    log_dir: str = "logs"
    log_file: str = "nids.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # JSON settings
    json_file: str = "nids.json"
    
    # Specific loggers
    uvicorn_level: str = "INFO"
    sqlalchemy_level: str = "WARNING"
    
    # Performance
    async_logging: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    
    # Metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    namespace: str = "nids"
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Alerting
    alerting_enabled: bool = False
    alert_webhook_url: str = ""
    
    # Performance monitoring
    profiling_enabled: bool = False
    memory_profiling: bool = False


@dataclass
class SecurityConfig:
    """Security configuration."""
    # API Security
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    
    # Authentication
    jwt_secret: str = ""  # Will be loaded from env
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    
    # Encryption
    encryption_key: str = ""  # Will be loaded from env
    
    # Request validation
    max_request_size: int = 1048576  # 1MB
    validate_content_type: bool = True
    
    # Security headers
    security_headers_enabled: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str = Environment.DEVELOPMENT.value
    debug: bool = True
    
    # Paths
    model_path: str = "models/autoencoder.pth"
    scaler_path: str = "models/scaler.pkl"
    config_path: str = "models/config.yaml"
    
    # Docker
    container_name: str = "nids-autoencoder"
    image_tag: str = "latest"
    
    # Kubernetes
    namespace: str = "nids"
    replicas: int = 3
    
    # Resource limits
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    cpu_request: str = "500m"
    memory_request: str = "1Gi"


@dataclass
class NIDSConfig:
    """Complete NIDS system configuration."""
    # Core components
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Infrastructure
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfigurationManager:
    """Enterprise configuration manager for NIDS system."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        override_env_vars: bool = True
    ):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, staging, production)
            override_env_vars: Whether to override with environment variables
        """
        self.config_path = Path(config_path) if config_path else None
        self.environment = environment or os.getenv("NIDS_ENVIRONMENT", "development")
        self.override_env_vars = override_env_vars
        
        # Initialize configuration
        self.config = NIDSConfig()
        self._load_configuration()
        
        logger.info(f"Configuration manager initialized for {self.environment} environment")
    
    def _load_configuration(self) -> None:
        """Load configuration from various sources."""
        # 1. Load default configuration
        self._load_defaults()
        
        # 2. Load from base config file
        if self.config_path and self.config_path.exists():
            self._load_from_file(self.config_path)
        
        # 3. Load environment-specific config
        self._load_environment_specific()
        
        # 4. Override with environment variables
        if self.override_env_vars:
            self._load_from_environment()
        
        # 5. Validate configuration
        self._validate_configuration()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config = NIDSConfig()
        logger.debug("Default configuration loaded")
    
    def _load_from_file(self, config_path: Path) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                return
            
            self._merge_config_data(config_data)
            logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_environment_specific(self) -> None:
        """Load environment-specific configuration."""
        if not self.config_path:
            return
            
        # Look for environment-specific config files
        base_path = self.config_path.parent
        env_config_path = base_path / f"{self.environment}.yaml"
        
        if env_config_path.exists():
            self._load_from_file(env_config_path)
            logger.info(f"Environment-specific config loaded: {env_config_path}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # Model configuration
            'NIDS_MODEL_INPUT_DIM': ('model', 'input_dim', int),
            'NIDS_MODEL_HIDDEN_DIMS': ('model', 'hidden_dims', self._parse_list_int),
            'NIDS_MODEL_LEARNING_RATE': ('training', 'learning_rate', float),
            'NIDS_MODEL_BATCH_SIZE': ('training', 'batch_size', int),
            'NIDS_MODEL_EPOCHS': ('training', 'epochs', int),
            
            # Data configuration
            'NIDS_DATA_SOURCE_PATH': ('data', 'source_path', str),
            'NIDS_DATA_NORMAL_CLASS': ('data', 'normal_class', str),
            
            # API configuration
            'NIDS_API_HOST': ('api', 'host', str),
            'NIDS_API_PORT': ('api', 'port', int),
            'NIDS_API_WORKERS': ('api', 'workers', int),
            
            # Database configuration
            'NIDS_DB_HOST': ('database', 'host', str),
            'NIDS_DB_PORT': ('database', 'port', int),
            'NIDS_DB_PASSWORD': ('database', 'password', str),
            
            # Security configuration
            'NIDS_JWT_SECRET': ('security', 'jwt_secret', str),
            'NIDS_ENCRYPTION_KEY': ('security', 'encryption_key', str),
            
            # Logging configuration
            'NIDS_LOG_LEVEL': ('logging', 'level', str),
            'NIDS_LOG_DIR': ('logging', 'log_dir', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    typed_value = type_func(value)
                    setattr(getattr(self.config, section), key, typed_value)
                    logger.debug(f"Updated {section}.{key} from environment")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    def _parse_list_int(self, value: str) -> List[int]:
        """Parse comma-separated string to list of integers."""
        return [int(x.strip()) for x in value.split(',')]
    
    def _merge_config_data(self, config_data: Dict[str, Any]) -> None:
        """Merge configuration data into current config.
        
        Args:
            config_data: Dictionary of configuration data
        """
        for section_name, section_data in config_data.items():
            if hasattr(self.config, section_name) and isinstance(section_data, dict):
                section = getattr(self.config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
    
    def _validate_configuration(self) -> None:
        """Validate the current configuration."""
        errors = []
        
        # Validate model configuration
        if self.config.model.input_dim <= 0:
            errors.append("Model input_dim must be positive")
        
        if len(self.config.model.hidden_dims) == 0:
            errors.append("Model must have at least one hidden layer")
        
        # Validate training configuration
        if self.config.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.config.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate data configuration
        if not self.config.data.source.get("path"):
            errors.append("Data source path cannot be empty")
        
        # Validate API configuration
        if not (1 <= self.config.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def save_configuration(
        self,
        output_path: Union[str, Path],
        format: ConfigFormat = ConfigFormat.YAML,
        include_metadata: bool = True
    ) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
            format: Output format (YAML or JSON)
            include_metadata: Whether to include metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update timestamp
        if include_metadata:
            self.config.updated_at = datetime.now().isoformat()
        
        config_dict = asdict(self.config)
        
        try:
            if format == ConfigFormat.YAML:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format == ConfigFormat.JSON:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary.
        
        Returns:
            Configuration summary dictionary
        """
        return {
            'environment': self.environment,
            'config_version': self.config.config_version,
            'model': {
                'input_dim': self.config.model.input_dim,
                'hidden_dims': self.config.model.hidden_dims,
                'architecture': f"{self.config.model.input_dim} → {' → '.join(map(str, self.config.model.hidden_dims))} → {self.config.model.output_dim}"
            },
            'training': {
                'epochs': self.config.training.epochs,
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate
            },
            'api': {
                'host': self.config.api.host,
                'port': self.config.api.port,
                'workers': self.config.api.workers
            },
            'data': {
                'source': self.config.data.source.get("path", ""),
                'features': len(self.config.data.features.get("columns", []))
            }
        }
    
    def create_environment_configs(self, base_dir: Path) -> None:
        """Create environment-specific configuration files.
        
        Args:
            base_dir: Base directory for configuration files
        """
        environments = {
            Environment.DEVELOPMENT: self._get_development_overrides(),
            Environment.STAGING: self._get_staging_overrides(),
            Environment.PRODUCTION: self._get_production_overrides(),
            Environment.TESTING: self._get_testing_overrides()
        }
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        for env, overrides in environments.items():
            env_config = NIDSConfig()
            
            # Apply overrides
            for section_name, section_data in overrides.items():
                if hasattr(env_config, section_name):
                    section = getattr(env_config, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            # Save environment config
            config_path = base_dir / f"{env.value}.yaml"
            temp_manager = ConfigurationManager()
            temp_manager.config = env_config
            temp_manager.save_configuration(config_path)
    
    def _get_development_overrides(self) -> Dict[str, Any]:
        """Get development environment overrides."""
        return {
            'deployment': {
                'environment': 'development',
                'debug': True
            },
            'logging': {
                'level': 'DEBUG',
                'console_enabled': True,
                'file_enabled': True
            },
            'api': {
                'docs_enabled': True,
                'workers': 1
            },
            'training': {
                'epochs': 50,  # Faster training for development
                'checkpoint_freq': 5
            }
        }
    
    def _get_staging_overrides(self) -> Dict[str, Any]:
        """Get staging environment overrides."""
        return {
            'deployment': {
                'environment': 'staging',
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'json_enabled': True
            },
            'api': {
                'docs_enabled': True,
                'workers': 2,
                'authentication_enabled': True
            },
            'monitoring': {
                'enabled': True,
                'prometheus_enabled': True
            }
        }
    
    def _get_production_overrides(self) -> Dict[str, Any]:
        """Get production environment overrides."""
        return {
            'deployment': {
                'environment': 'production',
                'debug': False
            },
            'logging': {
                'level': 'WARNING',
                'console_enabled': False,
                'json_enabled': True
            },
            'api': {
                'docs_enabled': False,
                'workers': 4,
                'authentication_enabled': True,
                'rate_limit_enabled': True
            },
            'security': {
                'cors_origins': [],  # Restrict CORS in production
                'security_headers_enabled': True
            },
            'monitoring': {
                'enabled': True,
                'prometheus_enabled': True,
                'alerting_enabled': True
            },
            'database': {
                'enabled': True,
                'pool_size': 20
            }
        }
    
    def _get_testing_overrides(self) -> Dict[str, Any]:
        """Get testing environment overrides."""
        return {
            'deployment': {
                'environment': 'testing',
                'debug': True
            },
            'logging': {
                'level': 'WARNING',  # Reduce noise in tests
                'console_enabled': False,
                'file_enabled': False
            },
            'training': {
                'epochs': 5,  # Very fast training for tests
                'batch_size': 16
            },
            'data': {
                'cache_enabled': False  # No caching in tests
            }
        }


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


# Global configuration instance
_global_config: Optional[ConfigurationManager] = None


def get_config() -> ConfigurationManager:
    """Get global configuration instance.
    
    Returns:
        Global configuration manager instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationManager()
    return _global_config


def initialize_config(
    config_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> ConfigurationManager:
    """Initialize global configuration.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        Initialized configuration manager
    """
    global _global_config
    _global_config = ConfigurationManager(config_path, environment)
    return _global_config


def load_config_from_dict(config_dict: Dict[str, Any]) -> ConfigurationManager:
    """Load configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration manager with loaded data
    """
    config_manager = ConfigurationManager()
    config_manager._merge_config_data(config_dict)
    config_manager._validate_configuration()
    return config_manager


# Convenience functions for backward compatibility
def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """Load configuration (backward compatibility).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration manager instance
    """
    return initialize_config(config_path)


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigurationManager()
    
    # Print configuration summary
    summary = config_manager.get_config_summary()
    print("Configuration Summary:")
    for section, data in summary.items():
        print(f"  {section}: {data}")
    
    # Save configuration
    config_manager.save_configuration("config/nids_config.yaml")
    
    # Create environment-specific configs
    config_manager.create_environment_configs(Path("config/environments"))
