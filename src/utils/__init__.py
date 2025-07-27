"""Utilities package for the NIDS system.

This package contains utility modules for various system operations:
- constants: System-wide constants and configuration values
- logger: Logging utilities and structured logging
- data_utils: Data validation, preprocessing, and manipulation utilities
- model_utils: Model management, metrics calculation, and training utilities
- api_utils: API request/response handling and validation utilities
- metrics_utils: Performance monitoring and metrics collection
- config_utils: Configuration management and environment handling
- config: Legacy configuration utilities (kept for compatibility)
"""

# Legacy imports (kept for backward compatibility)
from .config import Config, load_config, get_env_var
from .logger import get_logger, setup_logging, LoggerMixin, log_function_call, log_errors

# Import key classes and functions for easy access
from .constants import (
    ModelDefaults,
    DataConstants,
    APIConstants,
    LoggingConstants,
    PerformanceConstants,
    ThresholdMethods,
    ErrorMessages
)

from .data_utils import (
    validate_data_format,
    handle_missing_values,
    normalize_features,
    encode_categorical_features,
    split_normal_anomalous,
    DataValidator
)

from .model_utils import (
    ModelManager,
    ThresholdCalculator,
    MetricsCalculator,
    TrainingMonitor
)

from .api_utils import (
    ResponseFormatter,
    RequestValidator,
    DataConverter,
    RateLimiter,
    HealthChecker
)

from .metrics_utils import (
    MetricsCollector,
    PerformanceMonitor,
    AlertManager
)

from .config_utils import (
    ModelConfig,
    DataConfig,
    APIConfig,
    LoggingConfig,
    SystemConfig,
    ConfigManager,
    load_config as load_config_new
)

__all__ = [
    # Legacy exports (backward compatibility)
    "Config",
    "load_config", 
    "get_env_var",
    "get_logger",
    "setup_logging",
    "LoggerMixin",
    "log_function_call",
    "log_errors",
    
    # Constants
    'ModelDefaults',
    'DataConstants',
    'APIConstants',
    'LoggingConstants',
    'PerformanceConstants',
    'ThresholdMethods',
    'ErrorMessages',
    
    # Data utilities
    'validate_data_format',
    'handle_missing_values',
    'normalize_features',
    'encode_categorical_features',
    'split_normal_anomalous',
    'DataValidator',
    
    # Model utilities
    'ModelManager',
    'ThresholdCalculator',
    'MetricsCalculator',
    'TrainingMonitor',
    
    # API utilities
    'ResponseFormatter',
    'RequestValidator',
    'DataConverter',
    'RateLimiter',
    'HealthChecker',
    
    # Metrics utilities
    'MetricsCollector',
    'PerformanceMonitor',
    'AlertManager',
    
    # Configuration utilities
    'ModelConfig',
    'DataConfig',
    'APIConfig',
    'LoggingConfig',
    'SystemConfig',
    'ConfigManager',
    'load_config_new'
]

# Version information
__version__ = "1.0.0"
__author__ = "NIDS Development Team"
__description__ = "Utilities package for Network Intrusion Detection System"
