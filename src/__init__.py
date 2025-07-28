"""
Network Intrusion Detection System
A production-ready autoencoder-based anomaly detection system for network security.
"""

__version__ = "1.0.0"
__author__ = "Prakhar Gupta"
__email__ = "1998prakhargupta@gmail.com"
__license__ = "MIT"

# Core imports for easy access (with fallback handling)
try:
    from .models.autoencoder import AutoencoderModel
    _autoencoder_available = True
except ImportError as e:
    AutoencoderModel = None
    _autoencoder_available = False

try:
    from .core.predictor import AnomalyPredictor
    _predictor_available = True
except ImportError as e:
    AnomalyPredictor = None
    _predictor_available = False

try:
    from .core.trainer import ModelTrainer
    _trainer_available = True
except ImportError as e:
    ModelTrainer = None
    _trainer_available = False

try:
    from .data.processor import DataProcessor
    _processor_available = True
except ImportError as e:
    DataProcessor = None
    _processor_available = False

try:
    from .utils.simple_config import Config
    _config_available = True
except ImportError as e:
    Config = None
    _config_available = False

# Do not import API by default to avoid dependency issues
# API should be imported explicitly when needed

try:
    from .utils.logger import get_logger
    _logger_available = True
except ImportError as e:
    _logger_available = False
    def get_logger(name):
        import logging
        return logging.getLogger(name)

# Define __all__ based on successfully imported modules
__all__ = []
if _autoencoder_available and AutoencoderModel is not None:
    __all__.append("AutoencoderModel")
if _predictor_available and AnomalyPredictor is not None:
    __all__.append("AnomalyPredictor")
if _trainer_available and ModelTrainer is not None:
    __all__.append("ModelTrainer")
if _processor_available and DataProcessor is not None:
    __all__.append("DataProcessor")
if _config_available and Config is not None:
    __all__.append("Config")
if _logger_available:
    __all__.append("get_logger")
