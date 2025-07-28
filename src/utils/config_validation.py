"""
Configuration validation schemas using Pydantic.

This module provides comprehensive validation for all configuration sections
to ensure type safety and business rule compliance.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without pydantic
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda *args, **kwargs: lambda func: func
    root_validator = lambda *args, **kwargs: lambda func: func
    PYDANTIC_AVAILABLE = False


class EnvironmentType(str, Enum):
    """Valid environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeviceType(str, Enum):
    """Valid device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class ScalingMethod(str, Enum):
    """Valid scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class OptimizerType(str, Enum):
    """Valid optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Valid scheduler types."""
    NONE = "none"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_LR = "step_lr"


if PYDANTIC_AVAILABLE:
    class ModelArchitectureSchema(BaseModel):
        """Validation schema for model architecture configuration."""
        
        input_dim: int = Field(gt=0, description="Input dimension must be positive")
        hidden_dims: List[int] = Field(
            min_items=1, 
            description="Must have at least one hidden layer"
        )
        output_dim: Optional[int] = Field(
            default=None, 
            gt=0, 
            description="Output dimension must be positive if specified"
        )
        activation: str = Field(
            default="relu",
            regex="^(relu|leaky_relu|tanh|sigmoid|gelu|swish)$",
            description="Must be a valid activation function"
        )
        dropout_rate: float = Field(
            ge=0.0, 
            le=1.0, 
            description="Dropout rate must be between 0 and 1"
        )
        batch_norm: bool = True
        layer_norm: bool = False
        
        @validator('hidden_dims')
        def validate_hidden_dims(cls, v):
            if any(dim <= 0 for dim in v):
                raise ValueError("All hidden dimensions must be positive")
            return v
        
        @validator('output_dim')
        def validate_output_dim(cls, v, values):
            if v is None:
                return values.get('input_dim')
            return v
        
        class Config:
            extra = "forbid"


    class TrainingSchema(BaseModel):
        """Validation schema for training configuration."""
        
        epochs: int = Field(ge=1, le=10000, description="Epochs must be between 1 and 10000")
        batch_size: int = Field(ge=1, le=10000, description="Batch size must be between 1 and 10000")
        learning_rate: float = Field(gt=0.0, le=1.0, description="Learning rate must be positive and <= 1")
        weight_decay: float = Field(ge=0.0, description="Weight decay must be non-negative")
        optimizer: OptimizerType = OptimizerType.ADAM
        scheduler: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
        
        # Early stopping
        early_stopping_enabled: bool = True
        early_stopping_patience: int = Field(ge=1, description="Patience must be positive")
        early_stopping_min_delta: float = Field(ge=0.0, description="Min delta must be non-negative")
        early_stopping_monitor: str = Field(default="val_loss")
        
        # Validation
        validation_split: float = Field(gt=0.0, lt=1.0, description="Validation split must be between 0 and 1")
        validation_freq: int = Field(ge=1, description="Validation frequency must be positive")
        shuffle: bool = True
        
        # Reproducibility
        seed: int = Field(ge=0, description="Seed must be non-negative")
        deterministic: bool = True
        
        # Device
        device: DeviceType = DeviceType.AUTO
        mixed_precision: bool = False
        
        # Checkpointing
        checkpoint_enabled: bool = True
        checkpoint_freq: int = Field(ge=1, description="Checkpoint frequency must be positive")
        save_best_only: bool = True
        
        @validator('batch_size')
        def validate_batch_size_power_of_2(cls, v):
            # Recommend power of 2 for better performance
            if v & (v - 1) != 0:
                import warnings
                warnings.warn(f"Batch size {v} is not a power of 2, which may impact performance")
            return v
        
        class Config:
            extra = "forbid"


    class DataSchema(BaseModel):
        """Validation schema for data configuration."""
        
        source_path: str = Field(min_length=1, description="Source path cannot be empty")
        source_format: str = Field(regex="^(csv|json|parquet)$", description="Must be csv, json, or parquet")
        encoding: str = Field(default="utf-8")
        
        feature_columns: List[str] = Field(min_items=1, description="Must have at least one feature")
        target_column: str = Field(min_length=1, description="Target column cannot be empty")
        normal_class: str = Field(min_length=1, description="Normal class cannot be empty")
        anomaly_classes: List[str] = Field(min_items=1, description="Must have at least one anomaly class")
        
        # Preprocessing
        scaling_method: ScalingMethod = ScalingMethod.STANDARD
        feature_range: tuple = Field(default=(0, 1))
        handle_missing: str = Field(
            regex="^(mean|median|mode|drop|interpolate)$",
            description="Must be a valid missing value strategy"
        )
        missing_threshold: float = Field(ge=0.0, le=1.0, description="Missing threshold must be between 0 and 1")
        outlier_detection: str = Field(
            regex="^(iqr|zscore|isolation_forest|none)$",
            description="Must be a valid outlier detection method"
        )
        outlier_threshold: float = Field(gt=0.0, description="Outlier threshold must be positive")
        
        # Splitting
        train_ratio: float = Field(gt=0.0, lt=1.0, description="Train ratio must be between 0 and 1")
        val_ratio: float = Field(ge=0.0, lt=1.0, description="Validation ratio must be between 0 and 1")
        test_ratio: float = Field(ge=0.0, lt=1.0, description="Test ratio must be between 0 and 1")
        stratify: bool = True
        
        # Caching
        cache_enabled: bool = True
        cache_dir: str = Field(default="data/cache")
        
        @validator('feature_range')
        def validate_feature_range(cls, v):
            if len(v) != 2 or v[0] >= v[1]:
                raise ValueError("Feature range must be a tuple (min, max) with min < max")
            return v
        
        @root_validator
        def validate_split_ratios(cls, values):
            train_ratio = values.get('train_ratio', 0)
            val_ratio = values.get('val_ratio', 0)
            test_ratio = values.get('test_ratio', 0)
            
            total = train_ratio + val_ratio + test_ratio
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Split ratios must sum to 1.0, got {total}")
            
            return values
        
        @validator('anomaly_classes')
        def validate_anomaly_classes(cls, v, values):
            normal_class = values.get('normal_class')
            if normal_class and normal_class in v:
                raise ValueError("Normal class cannot be in anomaly classes list")
            return v
        
        class Config:
            extra = "forbid"


    class ThresholdSchema(BaseModel):
        """Validation schema for threshold configuration."""
        
        methods: List[str] = Field(min_items=1, description="Must have at least one threshold method")
        percentile_value: float = Field(ge=0.0, le=100.0, description="Percentile must be between 0 and 100")
        statistical_factor: float = Field(gt=0.0, description="Statistical factor must be positive")
        statistical_method: str = Field(
            regex="^(std|mad|iqr)$",
            description="Must be a valid statistical method"
        )
        roc_metric: str = Field(
            regex="^(youden|f1|balanced_accuracy)$",
            description="Must be a valid ROC metric"
        )
        target_precision: float = Field(gt=0.0, le=1.0, description="Target precision must be between 0 and 1")
        target_recall: float = Field(gt=0.0, le=1.0, description="Target recall must be between 0 and 1")
        default_method: str = Field(min_length=1, description="Default method cannot be empty")
        
        @validator('methods')
        def validate_methods(cls, v):
            valid_methods = {"percentile", "statistical", "roc_optimal", "precision_recall"}
            invalid_methods = set(v) - valid_methods
            if invalid_methods:
                raise ValueError(f"Invalid threshold methods: {invalid_methods}")
            return v
        
        @validator('default_method')
        def validate_default_method(cls, v, values):
            methods = values.get('methods', [])
            if v not in methods:
                raise ValueError(f"Default method '{v}' must be in methods list")
            return v
        
        class Config:
            extra = "forbid"


    class APISchema(BaseModel):
        """Validation schema for API configuration."""
        
        title: str = Field(min_length=1, description="API title cannot be empty")
        description: str = Field(min_length=1, description="API description cannot be empty")
        version: str = Field(regex=r"^\d+\.\d+\.\d+.*$", description="Must be valid semantic version")
        host: str = Field(regex=r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|localhost|[\w\.-]+)$")
        port: int = Field(ge=1, le=65535, description="Port must be between 1 and 65535")
        workers: int = Field(ge=1, le=32, description="Workers must be between 1 and 32")
        
        max_batch_size: int = Field(ge=1, description="Max batch size must be positive")
        timeout_seconds: int = Field(ge=1, description="Timeout must be positive")
        keepalive_timeout: int = Field(ge=0, description="Keepalive timeout must be non-negative")
        
        # Security
        authentication_enabled: bool = False
        api_key_header: str = Field(default="X-API-Key")
        allowed_origins: List[str] = Field(default_factory=list)
        
        # Rate limiting
        rate_limit_enabled: bool = True
        requests_per_minute: int = Field(ge=1, description="Requests per minute must be positive")
        burst_size: int = Field(ge=1, description="Burst size must be positive")
        
        # Monitoring
        metrics_enabled: bool = True
        health_check_enabled: bool = True
        request_logging: bool = True
        
        # Documentation
        docs_enabled: bool = True
        docs_url: str = Field(regex=r"^/.*", description="Docs URL must start with /")
        redoc_url: str = Field(regex=r"^/.*", description="ReDoc URL must start with /")
        
        class Config:
            extra = "forbid"


    class DatabaseSchema(BaseModel):
        """Validation schema for database configuration."""
        
        enabled: bool = False
        host: str = Field(default="localhost")
        port: int = Field(ge=1, le=65535, description="Port must be between 1 and 65535")
        database: str = Field(min_length=1, description="Database name cannot be empty")
        username: str = Field(min_length=1, description="Username cannot be empty")
        password: str = Field(default="")
        
        pool_size: int = Field(ge=1, le=100, description="Pool size must be between 1 and 100")
        max_overflow: int = Field(ge=0, le=100, description="Max overflow must be between 0 and 100")
        pool_timeout: int = Field(ge=1, description="Pool timeout must be positive")
        pool_recycle: int = Field(ge=0, description="Pool recycle must be non-negative")
        
        echo: bool = False
        echo_pool: bool = False
        
        class Config:
            extra = "forbid"


    class LoggingSchema(BaseModel):
        """Validation schema for logging configuration."""
        
        level: str = Field(
            regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
            description="Must be a valid log level"
        )
        format: str = Field(min_length=1, description="Log format cannot be empty")
        date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
        
        console_enabled: bool = True
        file_enabled: bool = True
        json_enabled: bool = False
        syslog_enabled: bool = False
        
        log_dir: str = Field(min_length=1, description="Log directory cannot be empty")
        log_file: str = Field(min_length=1, description="Log file cannot be empty")
        max_file_size: int = Field(ge=1024, description="Max file size must be at least 1KB")
        backup_count: int = Field(ge=0, le=100, description="Backup count must be between 0 and 100")
        
        json_file: str = Field(default="app.json")
        
        uvicorn_level: str = Field(
            regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
            description="Must be a valid log level"
        )
        sqlalchemy_level: str = Field(
            regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
            description="Must be a valid log level"
        )
        
        async_logging: bool = False
        
        class Config:
            extra = "forbid"


    class DeploymentSchema(BaseModel):
        """Validation schema for deployment configuration."""
        
        environment: EnvironmentType
        debug: bool = False
        
        model_path: str = Field(min_length=1, description="Model path cannot be empty")
        scaler_path: str = Field(min_length=1, description="Scaler path cannot be empty")
        config_path: str = Field(min_length=1, description="Config path cannot be empty")
        
        container_name: str = Field(regex=r"^[a-z0-9\-]+$", description="Invalid container name")
        image_tag: str = Field(min_length=1, description="Image tag cannot be empty")
        
        namespace: str = Field(regex=r"^[a-z0-9\-]+$", description="Invalid namespace")
        replicas: int = Field(ge=1, le=100, description="Replicas must be between 1 and 100")
        
        cpu_limit: str = Field(regex=r"^\d+m?$", description="Invalid CPU limit format")
        memory_limit: str = Field(regex=r"^\d+[GM]i$", description="Invalid memory limit format")
        cpu_request: str = Field(regex=r"^\d+m?$", description="Invalid CPU request format")
        memory_request: str = Field(regex=r"^\d+[GM]i$", description="Invalid memory request format")
        
        class Config:
            extra = "forbid"


    class NIDSConfigSchema(BaseModel):
        """Complete NIDS configuration validation schema."""
        
        model: ModelArchitectureSchema
        training: TrainingSchema
        data: DataSchema
        thresholds: ThresholdSchema
        api: APISchema
        database: DatabaseSchema
        logging: LoggingSchema
        deployment: DeploymentSchema
        
        config_version: str = Field(regex=r"^\d+\.\d+\.\d+$", description="Must be valid semantic version")
        created_at: str = Field(regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*$")
        updated_at: str = Field(regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*$")
        
        class Config:
            extra = "forbid"


def validate_config_dict(config_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if not PYDANTIC_AVAILABLE:
        return True, ["Pydantic not available, skipping validation"]
    
    try:
        NIDSConfigSchema(**config_dict)
        return True, []
    except Exception as e:
        error_messages = []
        if hasattr(e, 'errors'):
            for error in e.errors():
                field = '.'.join(str(x) for x in error['loc'])
                message = error['msg']
                error_messages.append(f"{field}: {message}")
        else:
            error_messages.append(str(e))
        
        return False, error_messages


def get_validation_schema() -> Optional[type]:
    """Get the validation schema class.
    
    Returns:
        NIDSConfigSchema class if pydantic is available, None otherwise
    """
    if PYDANTIC_AVAILABLE:
        return NIDSConfigSchema
    return None


# Fallback implementations when pydantic is not available
def validate_config_dict_fallback(config_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Basic validation without pydantic."""
    errors = []
    
    # Basic structure validation
    required_sections = ['model', 'training', 'data', 'api']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")
    
    # Basic value validation
    if 'training' in config_dict:
        training = config_dict['training']
        if 'epochs' in training and training['epochs'] <= 0:
            errors.append("training.epochs: Must be positive")
        if 'batch_size' in training and training['batch_size'] <= 0:
            errors.append("training.batch_size: Must be positive")
    
    return len(errors) == 0, errors


# Use appropriate validation function based on pydantic availability
if not PYDANTIC_AVAILABLE:
    validate_config_dict = validate_config_dict_fallback


__all__ = [
    'validate_config_dict',
    'get_validation_schema',
    'EnvironmentType',
    'DeviceType',
    'ScalingMethod',
    'OptimizerType',
    'SchedulerType'
]
