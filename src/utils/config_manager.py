"""
Simple Configuration Management System
=====================================

A simplified version compatible with Python 3.6+ that provides
enterprise-grade configuration management.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"
    PRODUCTION = "production"


class SimpleConfigManager:
    """
    Simple configuration manager with inheritance and validation.
    
    Features:
    - Hierarchical configuration inheritance (base -> environment -> local)
    - Environment variable substitution
    - Basic validation
    - Configuration hot-reloading
    """
    
    def __init__(self, config_dir="config", environment=None, validate_on_load=True):
        """Initialize the configuration manager."""
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.validate_on_load = validate_on_load
        
        # Internal state
        self._config = {}
        self._metadata = {
            'environment': self.environment,
            'loaded_at': datetime.now(),
            'config_files': [],
            'validation_errors': [],
            'warnings': [],
            'inheritance_chain': []
        }
        
        # Load initial configuration
        self.reload_config()
    
    def reload_config(self):
        """Reload configuration from files."""
        try:
            logger.info(f"Loading configuration for environment: {self.environment}")
            
            # Reset metadata
            self._metadata = {
                'environment': self.environment,
                'loaded_at': datetime.now(),
                'config_files': [],
                'validation_errors': [],
                'warnings': [],
                'inheritance_chain': []
            }
            
            # Load configuration with inheritance
            config = self._load_with_inheritance()
            
            # Apply environment variable substitution
            config = self._substitute_environment_variables(config)
            
            # Basic validation if enabled
            if self.validate_on_load:
                self._validate_configuration(config)
            
            # Store final configuration
            self._config = config
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def _load_with_inheritance(self):
        """Load configuration with inheritance chain: base -> environment -> local."""
        config = {}
        inheritance_chain = []
        
        # 1. Load base configuration
        base_file = self.config_dir / "base.yaml"
        if base_file.exists():
            base_config = self._load_yaml_file(base_file)
            config = self._deep_merge(config, base_config)
            inheritance_chain.append("base.yaml")
            self._metadata['config_files'].append(str(base_file))
        
        # 2. Load environment-specific configuration
        env_file = self.config_dir / f"{self.environment}.yaml"
        if env_file.exists():
            env_config = self._load_yaml_file(env_file)
            config = self._deep_merge(config, env_config)
            inheritance_chain.append(f"{self.environment}.yaml")
            self._metadata['config_files'].append(str(env_file))
        
        # 3. Load local overrides (optional)
        local_file = self.config_dir / "local.yaml"
        if local_file.exists():
            local_config = self._load_yaml_file(local_file)
            config = self._deep_merge(config, local_config)
            inheritance_chain.append("local.yaml")
            self._metadata['config_files'].append(str(local_file))
        
        self._metadata['inheritance_chain'] = inheritance_chain
        logger.debug(f"Configuration inheritance chain: {' -> '.join(inheritance_chain)}")
        
        return config
    
    def _load_yaml_file(self, file_path):
        """Load and parse a YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
                logger.debug(f"Loaded configuration from: {file_path}")
                return content
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        except Exception as e:
            error_msg = f"Error loading {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def _deep_merge(self, base, override):
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_environment_variables(self, config):
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
                        self._metadata['warnings'].append(warning)
                        return match.group(0)  # Return original if not found
                    
                    return env_value
                
                return re.sub(pattern, replace_var, value)
            
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            
            return value
        
        return substitute_value(config)
    
    def _validate_configuration(self, config):
        """Basic configuration validation."""
        validation_errors = []
        
        try:
            # Check required top-level sections
            required_sections = ['model', 'training', 'data', 'api', 'logging']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                validation_errors.append(f"Missing required sections: {missing_sections}")
            
            # Basic model validation
            if 'model' in config and 'architecture' in config['model']:
                arch = config['model']['architecture']
                
                if 'input_dim' in arch and (not isinstance(arch['input_dim'], int) or arch['input_dim'] <= 0):
                    validation_errors.append("model.architecture.input_dim must be a positive integer")
                
                if 'hidden_dims' in arch and not isinstance(arch['hidden_dims'], list):
                    validation_errors.append("model.architecture.hidden_dims must be a list")
            
            # Basic training validation
            if 'training' in config:
                training = config['training']
                
                if 'epochs' in training and (not isinstance(training['epochs'], int) or training['epochs'] <= 0):
                    validation_errors.append("training.epochs must be a positive integer")
                
                if 'batch_size' in training and (not isinstance(training['batch_size'], int) or training['batch_size'] <= 0):
                    validation_errors.append("training.batch_size must be a positive integer")
            
            # Production-specific validation
            if self.environment == 'production':
                if config.get('debug_mode', True):
                    validation_errors.append("Debug mode should be disabled in production")
                
                if config.get('logging', {}).get('level') == 'DEBUG':
                    validation_errors.append("Debug logging should not be enabled in production")
            
        except Exception as e:
            validation_errors.append(f"Configuration validation error: {str(e)}")
        
        # Store validation errors
        self._metadata['validation_errors'] = validation_errors
        
        # Raise exception if there are validation errors
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get(self, key, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
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
    
    def export_config(self, format_type="yaml"):
        """Export current configuration in specified format."""
        if format_type.lower() == "yaml":
            return yaml.dump(self._config, default_flow_style=False, sort_keys=True)
        elif format_type.lower() == "json":
            import json
            return json.dumps(self._config, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    @property
    def config(self):
        """Get the complete configuration dictionary."""
        return self._config.copy()
    
    @property
    def metadata(self):
        """Get configuration metadata."""
        return self._metadata.copy()
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key):
        """Check if configuration key exists."""
        return self.get(key) is not None


# Global configuration instance
_global_config = None


def get_config():
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = SimpleConfigManager()
    return _global_config


def init_config(config_dir="config", environment=None):
    """Initialize the global configuration manager."""
    global _global_config
    _global_config = SimpleConfigManager(
        config_dir=config_dir,
        environment=environment
    )
    return _global_config


# Convenience functions
def get_model_config():
    """Get model configuration."""
    return get_config().get('model', {})


def get_training_config():
    """Get training configuration."""
    return get_config().get('training', {})


def get_api_config():
    """Get API configuration."""
    return get_config().get('api', {})


def get_data_config():
    """Get data configuration."""
    return get_config().get('data', {})


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize configuration manager
        config_manager = SimpleConfigManager(
            config_dir="config",
            environment="development",
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
        print(f"Environment: {metadata['environment']}")
        print(f"Loaded at: {metadata['loaded_at']}")
        print(f"Config files: {metadata['config_files']}")
        print(f"Inheritance chain: {metadata['inheritance_chain']}")
        
        if metadata['validation_errors']:
            print(f"Validation errors: {metadata['validation_errors']}")
        
        if metadata['warnings']:
            print(f"Warnings: {metadata['warnings']}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
