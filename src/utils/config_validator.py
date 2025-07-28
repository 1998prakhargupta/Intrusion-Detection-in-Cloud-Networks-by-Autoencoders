"""
Configuration Validation Module
==============================

This module provides configuration validation utilities to reduce
complexity in the main configuration manager.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_model_config(model_config: Dict[str, Any], errors: List[str]) -> None:
        """Validate model configuration section."""
        if 'architecture' in model_config:
            arch = model_config['architecture']
            
            # Validate input_dim
            if 'input_dim' not in arch or not isinstance(arch['input_dim'], int) or arch['input_dim'] <= 0:
                errors.append("model.architecture.input_dim must be a positive integer")
            
            # Validate hidden_dims
            if 'hidden_dims' not in arch or not isinstance(arch['hidden_dims'], list):
                errors.append("model.architecture.hidden_dims must be a list")
            elif any(not isinstance(dim, int) or dim <= 0 for dim in arch['hidden_dims']):
                errors.append("All values in model.architecture.hidden_dims must be positive integers")
    
    @staticmethod
    def validate_training_config(training_config: Dict[str, Any], errors: List[str]) -> None:
        """Validate training configuration section."""
        # Validate epochs
        if 'epochs' in training_config and (not isinstance(training_config['epochs'], int) or training_config['epochs'] <= 0):
            errors.append("training.epochs must be a positive integer")
        
        # Validate batch_size
        if 'batch_size' in training_config and (not isinstance(training_config['batch_size'], int) or training_config['batch_size'] <= 0):
            errors.append("training.batch_size must be a positive integer")
        
        # Validate learning_rate
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr >= 1:
                errors.append("training.learning_rate must be a number between 0 and 1")
    
    @staticmethod
    def validate_api_config(api_config: Dict[str, Any], errors: List[str]) -> None:
        """Validate API configuration section."""
        if 'server' in api_config:
            server = api_config['server']
            
            # Validate port
            if 'port' in server:
                port = server['port']
                if not isinstance(port, int) or port < 1 or port > 65535:
                    errors.append("api.server.port must be an integer between 1 and 65535")
            
            # Validate workers
            if 'workers' in server:
                workers = server['workers']
                if not isinstance(workers, int) or workers < 1:
                    errors.append("api.server.workers must be a positive integer")
    
    @staticmethod
    def validate_production_config(config: Dict[str, Any], errors: List[str]) -> None:
        """Additional validation for production environment."""
        # Security checks
        if config.get('api', {}).get('security', {}).get('authentication_enabled') is False:
            errors.append("Authentication must be enabled in production")
        
        if config.get('logging', {}).get('level') == 'DEBUG':
            errors.append("Debug logging should not be enabled in production")
        
        # Performance checks
        if config.get('api', {}).get('server', {}).get('workers', 1) < 2:
            errors.append("Production should use multiple workers")
        
        # Monitoring checks
        if not config.get('monitoring', {}).get('enabled', False):
            errors.append("Monitoring must be enabled in production")
    
    @staticmethod
    def validate_required_sections(config: Dict[str, Any], errors: List[str]) -> None:
        """Validate that required configuration sections exist."""
        required_sections = ['model', 'training', 'data', 'api', 'logging']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            errors.append(f"Missing required sections: {missing_sections}")
