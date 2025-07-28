"""Configuration utilities for managing system settings."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict

from .constants import ModelDefaults, DataConstants, APIConstants, LoggingConstants, EnvVars
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    input_dim: int = ModelDefaults.INPUT_DIM
    hidden_dims: Optional[List[int]] = None
    learning_rate: float = ModelDefaults.LEARNING_RATE
    batch_size: int = ModelDefaults.BATCH_SIZE
    epochs: int = ModelDefaults.EPOCHS
    validation_split: float = ModelDefaults.VALIDATION_SPLIT
    early_stopping_patience: int = ModelDefaults.EARLY_STOPPING_PATIENCE
    dropout_rate: float = ModelDefaults.DROPOUT_RATE
    threshold_percentile: float = ModelDefaults.THRESHOLD_PERCENTILE
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = ModelDefaults.HIDDEN_DIMS.copy()


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_ratio: float = DataConstants.TRAIN_RATIO
    test_ratio: float = DataConstants.TEST_RATIO
    validation_ratio: float = DataConstants.VALIDATION_RATIO
    missing_value_strategy: str = DataConstants.MISSING_VALUE_STRATEGY
    categorical_encoding: str = DataConstants.CATEGORICAL_ENCODING
    normalization_method: str = DataConstants.NORMALIZATION_METHOD
    outlier_threshold: float = DataConstants.OUTLIER_THRESHOLD
    max_features: int = DataConstants.MAX_FEATURES


@dataclass
class APIConfig:
    """Configuration for API settings."""
    host: str = APIConstants.DEFAULT_HOST
    port: int = APIConstants.DEFAULT_PORT
    max_batch_size: int = APIConstants.MAX_BATCH_SIZE
    timeout_seconds: int = APIConstants.TIMEOUT_SECONDS
    rate_limit_requests: int = APIConstants.RATE_LIMIT_REQUESTS
    rate_limit_window: int = APIConstants.RATE_LIMIT_WINDOW
    enable_cors: bool = APIConstants.ENABLE_CORS
    enable_docs: bool = APIConstants.ENABLE_DOCS


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = LoggingConstants.DEFAULT_LEVEL
    format: str = LoggingConstants.FORMAT
    date_format: str = LoggingConstants.DATE_FORMAT
    max_file_size: int = LoggingConstants.MAX_FILE_SIZE
    backup_count: int = LoggingConstants.BACKUP_COUNT
    console_output: bool = True
    file_output: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""
    model: Optional[ModelConfig] = None
    data: Optional[DataConfig] = None
    api: Optional[APIConfig] = None
    logging: Optional[LoggingConfig] = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """Manager for system configuration."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional).
        """
        self.config_file = config_file
        self.config = SystemConfig()
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_environment()
    
    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file.
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'model' in config_data:
                model_data = config_data['model']
                self.config.model = ModelConfig(**model_data)
            
            if 'data' in config_data:
                data_data = config_data['data']
                self.config.data = DataConfig(**data_data)
            
            if 'api' in config_data:
                api_data = config_data['api']
                self.config.api = APIConfig(**api_data)
            
            if 'logging' in config_data:
                logging_data = config_data['logging']
                self.config.logging = LoggingConfig(**logging_data)
            
            logger.info(f"Configuration loaded from: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            logger.info("Using default configuration")
    
    def load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Model configuration
        if os.getenv(EnvVars.MODEL_INPUT_DIM):
            self.config.model.input_dim = int(os.getenv(EnvVars.MODEL_INPUT_DIM))
        
        if os.getenv(EnvVars.MODEL_LEARNING_RATE):
            self.config.model.learning_rate = float(os.getenv(EnvVars.MODEL_LEARNING_RATE))
        
        if os.getenv(EnvVars.MODEL_BATCH_SIZE):
            self.config.model.batch_size = int(os.getenv(EnvVars.MODEL_BATCH_SIZE))
        
        if os.getenv(EnvVars.MODEL_EPOCHS):
            self.config.model.epochs = int(os.getenv(EnvVars.MODEL_EPOCHS))
        
        # API configuration
        if os.getenv(EnvVars.API_HOST):
            self.config.api.host = os.getenv(EnvVars.API_HOST)
        
        if os.getenv(EnvVars.API_PORT):
            self.config.api.port = int(os.getenv(EnvVars.API_PORT))
        
        # Logging configuration
        if os.getenv(EnvVars.LOG_LEVEL):
            self.config.logging.level = os.getenv(EnvVars.LOG_LEVEL)
        
        logger.debug("Configuration updated from environment variables")
    
    def save_to_file(self, config_file: Path) -> None:
        """Save current configuration to JSON file.
        
        Args:
            config_file: Path to save configuration file.
        """
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                'model': asdict(self.config.model),
                'data': asdict(self.config.data),
                'api': asdict(self.config.api),
                'logging': asdict(self.config.logging)
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration.
        
        Returns:
            Model configuration.
        """
        return self.config.model
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration.
        
        Returns:
            Data configuration.
        """
        return self.config.data
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration.
        
        Returns:
            API configuration.
        """
        return self.config.api
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration.
        
        Returns:
            Logging configuration.
        """
        return self.config.logging
    
    def update_model_config(self, **kwargs) -> None:
        """Update model configuration.
        
        Args:
            **kwargs: Model configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
                logger.debug(f"Updated model config: {key} = {value}")
            else:
                logger.warning(f"Unknown model config parameter: {key}")
    
    def update_data_config(self, **kwargs) -> None:
        """Update data configuration.
        
        Args:
            **kwargs: Data configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config.data, key):
                setattr(self.config.data, key, value)
                logger.debug(f"Updated data config: {key} = {value}")
            else:
                logger.warning(f"Unknown data config parameter: {key}")
    
    def update_api_config(self, **kwargs) -> None:
        """Update API configuration.
        
        Args:
            **kwargs: API configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config.api, key):
                setattr(self.config.api, key, value)
                logger.debug(f"Updated API config: {key} = {value}")
            else:
                logger.warning(f"Unknown API config parameter: {key}")
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration.
        
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Validate model configuration
        if self.config.model.input_dim <= 0:
            errors.append("Model input_dim must be positive")
        
        if self.config.model.learning_rate <= 0:
            errors.append("Model learning_rate must be positive")
        
        if self.config.model.batch_size <= 0:
            errors.append("Model batch_size must be positive")
        
        if self.config.model.epochs <= 0:
            errors.append("Model epochs must be positive")
        
        # Validate data configuration
        total_ratio = (self.config.data.train_ratio + 
                      self.config.data.test_ratio + 
                      self.config.data.validation_ratio)
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate API configuration
        if not (1 <= self.config.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if self.config.api.max_batch_size <= 0:
            errors.append("API max_batch_size must be positive")
        
        return len(errors) == 0, errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary.
        
        Returns:
            Configuration summary dictionary.
        """
        return {
            'model': asdict(self.config.model),
            'data': asdict(self.config.data),
            'api': asdict(self.config.api),
            'logging': asdict(self.config.logging),
            'validation': {
                'is_valid': self.validate_configuration()[0],
                'errors': self.validate_configuration()[1]
            }
        }


def load_config(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Load configuration from file or environment.
    
    Args:
        config_file: Path to configuration file (optional).
        
    Returns:
        Configuration manager instance.
    """
    if config_file:
        config_file = Path(config_file)
    
    return ConfigManager(config_file)


def create_default_config_file(config_file: Path) -> None:
    """Create a default configuration file.
    
    Args:
        config_file: Path to create configuration file.
    """
    config_manager = ConfigManager()
    config_manager.save_to_file(config_file)
    logger.info(f"Default configuration file created: {config_file}")


def merge_configs(base_config: ConfigManager, 
                 override_config: Dict[str, Any]) -> ConfigManager:
    """Merge configuration with overrides.
    
    Args:
        base_config: Base configuration manager.
        override_config: Configuration overrides.
        
    Returns:
        New configuration manager with merged settings.
    """
    # Create a copy of base config
    merged_config = ConfigManager()
    merged_config.config = SystemConfig(
        model=ModelConfig(**asdict(base_config.config.model)),
        data=DataConfig(**asdict(base_config.config.data)),
        api=APIConfig(**asdict(base_config.config.api)),
        logging=LoggingConfig(**asdict(base_config.config.logging))
    )
    
    # Apply overrides
    if 'model' in override_config:
        merged_config.update_model_config(**override_config['model'])
    
    if 'data' in override_config:
        merged_config.update_data_config(**override_config['data'])
    
    if 'api' in override_config:
        merged_config.update_api_config(**override_config['api'])
    
    return merged_config
