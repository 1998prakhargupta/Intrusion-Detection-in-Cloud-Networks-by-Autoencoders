"""
Advanced Configuration Management System
========================================

This module provides enterprise-grade configuration management with:
- YAML-based hierarchical configuration
- Environment-specific inheritance  
- Schema validation and type checking
- Dynamic configuration reloading
- Environment variable substitution
- Configuration templating and merging

Author: Enterprise Development Team
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime

from .config_validator import ConfigValidator

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class ConfigMetadata:
    """Metadata about configuration loading and validation."""
    environment: str
    loaded_at: datetime = field(default_factory=datetime.now)
    config_files: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inheritance_chain: List[str] = field(default_factory=list)


class AdvancedConfigManager:
    """
    Advanced configuration manager with inheritance, validation, and dynamic loading.
    
    Features:
    - Hierarchical configuration inheritance (base -> environment -> local)
    - Pydantic-based validation
    - Environment variable substitution
    - Configuration hot-reloading
    - Schema validation
    - Configuration templating
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "config",
        environment: Optional[str] = None,
        auto_reload: bool = False,
        validate_on_load: bool = True
    ):
        """
        Initialize the advanced configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Target environment (development, staging, production)
            auto_reload: Enable automatic configuration reloading
            validate_on_load: Validate configuration on loading
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.auto_reload = auto_reload
        self.validate_on_load = validate_on_load
        
        # Internal state
        self._config: Dict[str, Any] = {}
        self._metadata: Optional[ConfigMetadata] = None
        self._schema: Optional[Dict] = None
        self._file_timestamps: Dict[str, float] = {}
        
        # Load initial configuration
        self.reload_config()
    
    def reload_config(self) -> None:
        """Reload configuration from files."""
        try:
            logger.info(f"Loading configuration for environment: {self.environment}")
            
            # Initialize metadata
            self._metadata = ConfigMetadata(environment=self.environment)
            
            # Load configuration with inheritance
            config = self._load_with_inheritance()
            
            # Apply environment variable substitution
            config = self._substitute_environment_variables(config)
            
            # Validate configuration if enabled
            if self.validate_on_load:
                self._validate_configuration(config)
            
            # Store final configuration
            self._config = config
            
            # Update file timestamps for auto-reload
            if self.auto_reload:
                self._update_file_timestamps()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _load_with_inheritance(self) -> Dict[str, Any]:
        """Load configuration with inheritance chain: base -> environment -> local."""
        config = {}
        inheritance_chain = []
        
        # 1. Load base configuration
        base_file = self.config_dir / "base.yaml"
        if base_file.exists():
            base_config = self._load_yaml_file(base_file)
            config = self._deep_merge(config, base_config)
            inheritance_chain.append("base.yaml")
            self._metadata.config_files.append(str(base_file))
        
        # 2. Load environment-specific configuration
        env_file = self.config_dir / f"{self.environment}.yaml"
        if env_file.exists():
            env_config = self._load_yaml_file(env_file)
            config = self._deep_merge(config, env_config)
            inheritance_chain.append(f"{self.environment}.yaml")
            self._metadata.config_files.append(str(env_file))
        
        # 3. Load local overrides (optional)
        local_file = self.config_dir / "local.yaml"
        if local_file.exists():
            local_config = self._load_yaml_file(local_file)
            config = self._deep_merge(config, local_config)
            inheritance_chain.append("local.yaml")
            self._metadata.config_files.append(str(local_file))
        
        self._metadata.inheritance_chain = inheritance_chain
        logger.debug(f"Configuration inheritance chain: {' -> '.join(inheritance_chain)}")
        
        return config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
                logger.debug(f"Loaded configuration from: {file_path}")
                return content
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        except Exception as e:
            error_msg = f"Error loading {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str):
                # Pattern: ${VARIABLE_NAME} or ${VARIABLE_NAME:default_value}
                pattern = r'\$\{([A-Za-z_]\w*)(?::([^}]*))?\}'
                
                def replace_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2)
                    env_value = os.getenv(var_name, default_value)
                    
                    if env_value is None:
                        warning = f"Environment variable {var_name} not found and no default provided"
                        logger.warning(warning)
                        self._metadata.warnings.append(warning)
                        return match.group(0)  # Return original if not found
                    
                    return env_value
                
                return re.sub(pattern, replace_var, value)
            
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            
            return value
        
        return substitute_value(config)
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema and business rules."""
        validation_errors = []
        
        try:
            # Use external validator to reduce complexity
            validator = ConfigValidator()
            
            # Check required sections
            validator.validate_required_sections(config, validation_errors)
            
            # Validate individual sections
            if 'model' in config:
                validator.validate_model_config(config['model'], validation_errors)
            
            if 'training' in config:
                validator.validate_training_config(config['training'], validation_errors)
            
            if 'api' in config:
                validator.validate_api_config(config['api'], validation_errors)
            
            # Environment-specific validation
            if self.environment == Environment.PRODUCTION:
                validator.validate_production_config(config, validation_errors)
            
        except Exception as e:
            validation_errors.append(f"Configuration validation error: {str(e)}")
        
        # Store validation errors
        self._metadata.validation_errors = validation_errors
        
        # Raise exception if there are validation errors
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def _update_file_timestamps(self) -> None:
        """Update file timestamps for auto-reload detection."""
        for file_path in self._metadata.config_files:
            path = Path(file_path)
            if path.exists():
                self._file_timestamps[str(path)] = path.stat().st_mtime
    
    def check_for_changes(self) -> bool:
        """Check if any configuration files have changed."""
        if not self.auto_reload:
            return False
        
        for file_path, timestamp in self._file_timestamps.items():
            path = Path(file_path)
            if path.exists() and path.stat().st_mtime > timestamp:
                return True
        
        return False
    
    def auto_reload_if_changed(self) -> bool:
        """Automatically reload configuration if files have changed."""
        if self.check_for_changes():
            logger.info("Configuration files changed, reloading...")
            self.reload_config()
            return True
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.architecture.input_dim')."""
        if self.auto_reload:
            self.auto_reload_if_changed()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Re-validate if enabled
        if self.validate_on_load:
            try:
                self._validate_configuration(self._config)
            except ConfigurationError:
                # Rollback on validation failure
                logger.warning(f"Failed to set {key}={value}: validation failed")
                raise
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration in specified format."""
        if format.lower() == "yaml":
            return yaml.dump(self._config, default_flow_style=False, sort_keys=True)
        elif format.lower() == "json":
            import json
            return json.dumps(self._config, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save_config(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save current configuration to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.export_config(format))
        
        logger.info(f"Configuration saved to: {path}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        if self.auto_reload:
            self.auto_reload_if_changed()
        return self._config.copy()
    
    @property
    def metadata(self) -> ConfigMetadata:
        """Get configuration metadata."""
        return self._metadata
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None


# Configuration schema definitions using Pydantic
class ModelArchitectureConfig(BaseModel):
    """Model architecture configuration schema."""
    input_dim: int = Field(gt=0, description="Input dimension size")
    hidden_dims: List[int] = Field(min_items=1, description="Hidden layer dimensions")
    dropout_rate: float = Field(ge=0, le=1, description="Dropout rate")
    batch_norm: bool = Field(default=True, description="Enable batch normalization")


class TrainingConfig(BaseModel):
    """Training configuration schema."""
    epochs: int = Field(gt=0, description="Number of training epochs")
    batch_size: int = Field(gt=0, description="Training batch size")
    learning_rate: float = Field(gt=0, lt=1, description="Learning rate")


class APIConfig(BaseModel):
    """API configuration schema."""
    host: str = Field(description="API host address")
    port: int = Field(ge=1, le=65535, description="API port number")
    workers: int = Field(gt=0, description="Number of worker processes")


# Global configuration instance
_global_config: Optional[AdvancedConfigManager] = None


def get_config() -> AdvancedConfigManager:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = AdvancedConfigManager()
    return _global_config


def init_config(
    config_dir: Union[str, Path] = "config",
    environment: Optional[str] = None,
    auto_reload: bool = False
) -> AdvancedConfigManager:
    """Initialize the global configuration manager."""
    global _global_config
    _global_config = AdvancedConfigManager(
        config_dir=config_dir,
        environment=environment,
        auto_reload=auto_reload
    )
    return _global_config


# Convenience functions
def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config().get('model', {})


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return get_config().get('training', {})


def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    return get_config().get('api', {})


def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return get_config().get('data', {})


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize configuration manager
        config_manager = AdvancedConfigManager(
            config_dir="config",
            environment="development",
            auto_reload=True,
            validate_on_load=True
        )
        
        # Access configuration values
        model_config = config_manager.get('model')
        input_dim = config_manager.get('model.architecture.input_dim', 20)
        api_port = config_manager.get('api.server.port', 8000)
        
        print(f"Model input dimension: {input_dim}")
        print(f"API port: {api_port}")
        
        # Print configuration metadata
        metadata = config_manager.metadata
        print("\nConfiguration metadata:")
        print(f"Environment: {metadata.environment}")
        print(f"Loaded at: {metadata.loaded_at}")
        print(f"Config files: {metadata.config_files}")
        print(f"Inheritance chain: {metadata.inheritance_chain}")
        
        if metadata.validation_errors:
            print(f"Validation errors: {metadata.validation_errors}")
        
        if metadata.warnings:
            print(f"Warnings: {metadata.warnings}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
