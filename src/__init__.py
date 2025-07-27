"""
Network Intrusion Detection System
A production-ready autoencoder-based anomaly detection system for network security.
"""

__version__ = "1.0.0"
__author__ = "Prakhar Gupta"
__email__ = "1998prakhargupta@gmail.com"
__license__ = "MIT"

# Core imports for easy access
from src.core.autoencoder import AutoencoderModel
from src.core.predictor import AnomalyPredictor
from src.core.trainer import ModelTrainer
from src.data.processor import DataProcessor
from src.utils.config import Config
from src.utils.logger import get_logger

__all__ = [
    "AutoencoderModel",
    "AnomalyPredictor", 
    "ModelTrainer",
    "DataProcessor",
    "Config",
    "get_logger",
]
