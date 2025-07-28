"""Constants and configuration values for the NIDS system."""

from enum import Enum
from pathlib import Path
import os

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data file paths
CIDDS_DATASET_FILE = "CIDDS-001-external-week3_1.csv"
DEFAULT_DATA_PATH = DATA_DIR / CIDDS_DATASET_FILE

# Model file patterns
MODEL_FILE_PATTERNS = {
    "pytorch": "*.pth",
    "numpy": "*.pkl",
    "scaler": "*_scaler.pkl",
    "autoencoder": "autoencoder_*.pth"
}

# Default model parameters
class ModelDefaults:
    """Default model architecture and training parameters."""
    
    # Autoencoder architecture
    INPUT_DIM = 11
    HIDDEN_DIMS = [64, 32, 16, 8]
    ACTIVATION = "relu"
    DROPOUT_RATE = 0.1
    
    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    VALIDATION_SPLIT = 0.2
    
    # Threshold parameters
    THRESHOLD_PERCENTILE = 95.0
    
    # Scaling parameters
    FEATURE_RANGE = (0, 1)
    SCALER_TYPE = "standard"

# Data processing constants
class DataConstants:
    """Constants for data processing."""
    
    # Data file path
    DATA_FILE_PATH = DEFAULT_DATA_PATH
    
    # Column names to exclude from features
    EXCLUDED_COLUMNS = ["class", "attackType", "attackID", "attackDescription"]
    
    # Normal traffic identifiers
    NORMAL_CLASS_NAMES = ["normal"]
    
    # Missing value handling
    MISSING_VALUE_STRATEGY = "mean"  # "mean", "median", "mode", "drop"
    
    # Train/test split ratios
    TRAIN_RATIO = 0.8
    VALIDATION_RATIO = 0.2
    TEST_RATIO = 0.15  # Added missing constant
    
    # Data processing parameters
    CATEGORICAL_ENCODING = "label"  # Added missing constant
    NORMALIZATION_METHOD = "standard"  # Added missing constant
    OUTLIER_THRESHOLD = 3.0  # Added missing constant
    MAX_FEATURES = 1000  # Added missing constant
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

# Threshold methods
class ThresholdMethods(Enum):
    """Enumeration of threshold calculation methods."""
    
    PERCENTILE = "percentile"
    STATISTICAL = "statistical"
    ROC_OPTIMAL = "roc_optimal"
    YOUDEN = "youden"

# Default threshold parameters
class ThresholdDefaults:
    """Default parameters for threshold calculations."""
    
    PERCENTILE_VALUE = 95.0
    STATISTICAL_N_STD = 2.0
    DEFAULT_METHOD = ThresholdMethods.PERCENTILE.value

# API constants
class APIConstants:
    """Constants for the REST API."""
    
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8000
    DEFAULT_WORKERS = 4
    DEFAULT_TIMEOUT = 30
    TIMEOUT_SECONDS = 30  # Added missing constant
    
    # Request limits
    MAX_BATCH_SIZE = 1000  # Added missing constant
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 100
    RATE_LIMIT_REQUESTS = 100  # Added missing constant
    RATE_LIMIT_WINDOW = 60  # seconds
    
    # CORS settings
    ENABLE_CORS = True  # Added missing constant
    ENABLE_DOCS = True  # Added missing constant
    
    # Response codes
    SUCCESS_CODE = 200
    ERROR_CODE = 500
    NOT_FOUND_CODE = 404
    BAD_REQUEST_CODE = 400

# Logging constants
class LoggingConstants:
    """Constants for logging configuration."""
    
    DEFAULT_LEVEL = "INFO"
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"  # Added missing constant
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # Added missing constant
    JSON_FORMAT = True
    
    # File rotation
    MAX_BYTES = 10485760  # 10MB
    MAX_FILE_SIZE = 10485760  # 10MB - Added missing constant
    BACKUP_COUNT = 5
    
    # Log file names
    MAIN_LOG_FILE = "nids.log"
    ERROR_LOG_FILE = "nids_errors.log"
    ACCESS_LOG_FILE = "nids_access.log"

# Performance constants
class PerformanceConstants:
    """Constants for performance evaluation."""
    
    # Metrics collection
    METRICS_HISTORY_SIZE = 1000  # Added missing constant
    
    # Metrics calculation
    ZERO_DIVISION_REPLACEMENT = 0.0
    
    # AUC calculation
    AUC_SAMPLE_POINTS = 100
    
    # Performance thresholds
    EXCELLENT_AUC = 0.9
    GOOD_AUC = 0.8
    FAIR_AUC = 0.7
    
    EXCELLENT_FPR = 0.05
    GOOD_FPR = 0.1
    
    EXCELLENT_TPR = 0.8
    GOOD_TPR = 0.6

# Environment variable names
class EnvVars:
    """Environment variable names."""
    
    # Paths
    DATA_PATH = "NIDS_DATA_PATH"
    MODEL_PATH = "NIDS_MODEL_PATH"
    CONFIG_PATH = "NIDS_CONFIG_PATH"
    LOG_PATH = "NIDS_LOG_PATH"
    
    # API configuration
    API_HOST = "NIDS_API_HOST"
    API_PORT = "NIDS_API_PORT"
    API_WORKERS = "NIDS_API_WORKERS"
    API_KEY = "NIDS_API_KEY"
    
    # Model configuration
    MODEL_TYPE = "NIDS_MODEL_TYPE"
    THRESHOLD_METHOD = "NIDS_THRESHOLD_METHOD"
    
    # Logging
    LOG_LEVEL = "NIDS_LOG_LEVEL"
    LOG_FORMAT = "NIDS_LOG_FORMAT"
    
    # Database (future use)
    DATABASE_URL = "NIDS_DATABASE_URL"
    REDIS_URL = "NIDS_REDIS_URL"

# File extensions
class FileExtensions:
    """Common file extensions."""
    
    PYTORCH_MODEL = ".pth"
    NUMPY_MODEL = ".pkl"
    CONFIG_FILE = ".yaml"
    LOG_FILE = ".log"
    CSV_FILE = ".csv"
    JSON_FILE = ".json"

# Error messages
class ErrorMessages:
    """Standard error messages."""
    
    MODEL_NOT_LOADED = "Model not loaded. Please load a model first."
    SCALER_NOT_LOADED = "Data scaler not loaded. Please load a scaler first."
    INVALID_THRESHOLD_METHOD = "Invalid threshold method. Choose from: {methods}"
    FILE_NOT_FOUND = "File not found: {path}"
    INVALID_DATA_FORMAT = "Invalid data format. Expected numpy array or pandas DataFrame."
    INSUFFICIENT_DATA = "Insufficient data for training. Need at least {min_samples} samples."
    TRAINING_FAILED = "Model training failed: {error}"
    PREDICTION_FAILED = "Prediction failed: {error}"

# Success messages
class SuccessMessages:
    """Standard success messages."""
    
    MODEL_LOADED = "Model loaded successfully from: {path}"
    MODEL_SAVED = "Model saved successfully to: {path}"
    TRAINING_COMPLETED = "Training completed successfully. Final loss: {loss:.6f}"
    PREDICTION_COMPLETED = "Prediction completed. Processed {samples} samples."
    THRESHOLD_OPTIMIZED = "Thresholds optimized successfully."

# Feature engineering constants
class FeatureConstants:
    """Constants for feature engineering."""
    
    # Categorical encoding
    ENCODING_METHODS = ["factorize", "onehot", "target"]
    DEFAULT_ENCODING = "factorize"
    
    # Feature selection
    MIN_VARIANCE_THRESHOLD = 0.0
    MAX_CORRELATION_THRESHOLD = 0.95
    
    # Outlier detection
    OUTLIER_Z_THRESHOLD = 3.0
    OUTLIER_IQR_MULTIPLIER = 1.5

# Validation constants
class ValidationConstants:
    """Constants for data validation."""
    
    MIN_SAMPLES_REQUIRED = 100
    MIN_FEATURES_REQUIRED = 5
    MAX_MISSING_RATIO = 0.3
    
    # Data type validation
    NUMERIC_DTYPES = ["int64", "float64", "int32", "float32"]
    CATEGORICAL_DTYPES = ["object", "category"]

def get_env_var(var_name: str, default_value=None, var_type=str):
    """Get environment variable with type conversion and default value.
    
    Args:
        var_name: Environment variable name.
        default_value: Default value if variable not found.
        var_type: Type to convert the value to.
        
    Returns:
        Environment variable value with proper type.
    """
    value = os.getenv(var_name, default_value)
    
    if value is None:
        return None
    
    if var_type == bool:
        return str(value).lower() in ("true", "1", "yes", "on")
    elif var_type == int:
        return int(value)
    elif var_type == float:
        return float(value)
    else:
        return str(value)

def get_data_path() -> Path:
    """Get the data file path from environment or default."""
    env_path = get_env_var(EnvVars.DATA_PATH)
    if env_path:
        return Path(env_path)
    return DEFAULT_DATA_PATH

def get_model_path() -> Path:
    """Get the model directory path from environment or default."""
    env_path = get_env_var(EnvVars.MODEL_PATH)
    if env_path:
        return Path(env_path)
    return MODELS_DIR

def get_log_level() -> str:
    """Get the log level from environment or default."""
    return get_env_var(EnvVars.LOG_LEVEL, LoggingConstants.DEFAULT_LEVEL)

def get_api_config() -> dict:
    """Get API configuration from environment variables."""
    return {
        "host": get_env_var(EnvVars.API_HOST, APIConstants.DEFAULT_HOST),
        "port": get_env_var(EnvVars.API_PORT, APIConstants.DEFAULT_PORT, int),
        "workers": get_env_var(EnvVars.API_WORKERS, APIConstants.DEFAULT_WORKERS, int),
        "api_key": get_env_var(EnvVars.API_KEY)
    }
