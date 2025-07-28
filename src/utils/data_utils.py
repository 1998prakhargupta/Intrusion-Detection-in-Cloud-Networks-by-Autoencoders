"""Utility functions for data validation and preprocessing."""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path

from .constants import (
    DataConstants, ValidationConstants, FeatureConstants, 
    ErrorMessages, SuccessMessages
)
from .logger import get_logger

logger = get_logger(__name__)


def validate_data_format(data: Union[np.ndarray, pd.DataFrame]) -> bool:
    """Validate that data is in expected format.
    
    Args:
        data: Input data to validate.
        
    Returns:
        True if data format is valid.
        
    Raises:
        ValueError: If data format is invalid.
    """
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise ValueError(ErrorMessages.INVALID_DATA_FORMAT)
    
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional array")
        if data.shape[0] < ValidationConstants.MIN_SAMPLES_REQUIRED:
            raise ValueError(
                ErrorMessages.INSUFFICIENT_DATA.format(
                    min_samples=ValidationConstants.MIN_SAMPLES_REQUIRED
                )
            )
        if data.shape[1] < ValidationConstants.MIN_FEATURES_REQUIRED:
            raise ValueError(f"Need at least {ValidationConstants.MIN_FEATURES_REQUIRED} features")
    
    elif isinstance(data, pd.DataFrame):
        if len(data) < ValidationConstants.MIN_SAMPLES_REQUIRED:
            raise ValueError(
                ErrorMessages.INSUFFICIENT_DATA.format(
                    min_samples=ValidationConstants.MIN_SAMPLES_REQUIRED
                )
            )
        if len(data.columns) < ValidationConstants.MIN_FEATURES_REQUIRED:
            raise ValueError(f"Need at least {ValidationConstants.MIN_FEATURES_REQUIRED} features")
    
    return True


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate and convert file path.
    
    Args:
        file_path: Path to validate.
        
    Returns:
        Validated Path object.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(ErrorMessages.FILE_NOT_FOUND.format(path=path))
    return path


def check_missing_values(data: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing values in DataFrame.
    
    Args:
        data: DataFrame to check.
        
    Returns:
        Dictionary with missing value statistics.
    """
    missing_stats = {
        'total_missing': data.isnull().sum().sum(),
        'missing_by_column': data.isnull().sum().to_dict(),
        'missing_ratio_by_column': (data.isnull().sum() / len(data)).to_dict(),
        'columns_with_missing': data.columns[data.isnull().any()].tolist()
    }
    
    total_cells = len(data) * len(data.columns)
    missing_stats['overall_missing_ratio'] = missing_stats['total_missing'] / total_cells
    
    return missing_stats


def handle_missing_values(data: pd.DataFrame, 
                         strategy: str = DataConstants.MISSING_VALUE_STRATEGY) -> pd.DataFrame:
    """Handle missing values in DataFrame.
    
    Args:
        data: DataFrame with potential missing values.
        strategy: Strategy for handling missing values.
        
    Returns:
        DataFrame with missing values handled.
    """
    data_clean = data.copy()
    
    if strategy == "drop":
        return _handle_missing_drop(data_clean)
    elif strategy == "mean":
        return _handle_missing_mean(data_clean)
    elif strategy == "median":
        return _handle_missing_median(data_clean)
    else:
        logger.warning(f"Unknown strategy: {strategy}. Returning original data.")
        return data_clean


def _handle_missing_drop(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by dropping rows."""
    data_clean = data.dropna()
    logger.info(f"Dropped rows with missing values. Shape: {data_clean.shape}")
    return data_clean


def _handle_missing_mean(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling with mean/mode."""
    data_clean = data.copy()
    
    # Fill numeric with mean
    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data_clean[col].isnull().any():
            mean_val = data_clean[col].mean()
            data_clean[col].fillna(mean_val, inplace=True)
            logger.debug(f"Filled {col} with mean: {mean_val}")
    
    # Fill categorical with mode
    categorical_columns = data_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data_clean[col].isnull().any():
            mode_val = data_clean[col].mode()[0] if not data_clean[col].mode().empty else 'unknown'
            data_clean[col].fillna(mode_val, inplace=True)
            logger.debug(f"Filled {col} with mode: {mode_val}")
    
    return data_clean


def _handle_missing_median(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling with median."""
    data_clean = data.copy()
    
    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data_clean[col].isnull().any():
            median_val = data_clean[col].median()
            data_clean[col].fillna(median_val, inplace=True)
            logger.debug(f"Filled {col} with median: {median_val}")
    
    return data_clean


def encode_categorical_features(data: pd.DataFrame, 
                              method: str = FeatureConstants.DEFAULT_ENCODING) -> pd.DataFrame:
    """Encode categorical features to numeric.
    
    Args:
        data: DataFrame with categorical features.
        method: Encoding method to use.
        
    Returns:
        DataFrame with encoded features.
    """
    data_encoded = data.copy()
    categorical_columns = data_encoded.select_dtypes(include=['object']).columns
    
    if method == "factorize":
        for col in categorical_columns:
            data_encoded[col] = pd.factorize(data_encoded[col])[0]
            logger.debug(f"Factorized {col}: {data_encoded[col].nunique()} unique values")
    
    elif method == "onehot":
        data_encoded = pd.get_dummies(data_encoded, columns=categorical_columns, drop_first=True)
        logger.debug(f"One-hot encoded {len(categorical_columns)} categorical columns")
    
    return data_encoded


def remove_low_variance_features(data: pd.DataFrame, 
                                threshold: float = FeatureConstants.MIN_VARIANCE_THRESHOLD) -> pd.DataFrame:
    """Remove features with low variance.
    
    Args:
        data: DataFrame to process.
        threshold: Minimum variance threshold.
        
    Returns:
        DataFrame with low-variance features removed.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    variance = numeric_data.var()
    
    low_variance_cols = variance[variance <= threshold].index.tolist()
    
    if low_variance_cols:
        logger.info(f"Removing {len(low_variance_cols)} low-variance columns: {low_variance_cols}")
        data_filtered = data.drop(columns=low_variance_cols)
    else:
        data_filtered = data.copy()
    
    return data_filtered


def detect_outliers(data: np.ndarray, method: str = "zscore", threshold: float = 3.0) -> np.ndarray:
    """Detect outliers in numeric data.
    
    Args:
        data: Numeric data array.
        method: Outlier detection method ('zscore' or 'iqr').
        threshold: Threshold for outlier detection.
        
    Returns:
        Boolean array indicating outliers.
    """
    if method == "zscore":
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        outliers = np.any(z_scores > threshold, axis=1)
    
    elif method == "iqr":
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers


def split_normal_anomalous(data: pd.DataFrame, 
                          class_column: str = "class",
                          normal_classes: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into normal and anomalous samples.
    
    Args:
        data: DataFrame with class labels.
        class_column: Name of the class column.
        normal_classes: List of class names considered normal.
        
    Returns:
        Tuple of (normal_data, anomalous_data).
    """
    if normal_classes is None:
        normal_classes = DataConstants.NORMAL_CLASS_NAMES
    
    # Find normal class in data
    available_classes = data[class_column].unique()
    normal_class = None
    
    for norm_class in normal_classes:
        if norm_class in available_classes:
            normal_class = norm_class
            break
    
    if normal_class is None:
        # Use most frequent class as normal
        normal_class = data[class_column].mode()[0]
        logger.warning(f"No predefined normal class found. Using most frequent: {normal_class}")
    
    normal_data = data[data[class_column] == normal_class].copy()
    anomalous_data = data[data[class_column] != normal_class].copy()
    
    logger.info(f"Split data: {len(normal_data)} normal, {len(anomalous_data)} anomalous samples")
    
    return normal_data, anomalous_data


def extract_features(data: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
    """Extract feature columns from DataFrame.
    
    Args:
        data: Input DataFrame.
        exclude_columns: Columns to exclude from features.
        
    Returns:
        DataFrame with only feature columns.
    """
    if exclude_columns is None:
        exclude_columns = DataConstants.EXCLUDED_COLUMNS
    
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    features = data[feature_columns].copy()
    
    logger.info(f"Extracted {len(feature_columns)} feature columns")
    logger.debug(f"Feature columns: {feature_columns}")
    
    return features


def validate_feature_consistency(train_features: pd.DataFrame, 
                                test_features: pd.DataFrame) -> bool:
    """Validate that train and test features are consistent.
    
    Args:
        train_features: Training features.
        test_features: Test features.
        
    Returns:
        True if features are consistent.
        
    Raises:
        ValueError: If features are inconsistent.
    """
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test features have different columns")
    
    if train_features.dtypes.to_dict() != test_features.dtypes.to_dict():
        logger.warning("Train and test features have different data types")
    
    return True


def create_feature_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for features.
    
    Args:
        data: DataFrame to summarize.
        
    Returns:
        Dictionary with feature summary statistics.
    """
    summary = {
        'shape': data.shape,
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': check_missing_values(data),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Add statistics for numeric columns
    if summary['numeric_columns']:
        numeric_stats = data[summary['numeric_columns']].describe()
        summary['numeric_statistics'] = numeric_stats.to_dict()
    
    # Add value counts for categorical columns
    if summary['categorical_columns']:
        categorical_stats = {}
        for col in summary['categorical_columns']:
            categorical_stats[col] = {
                'unique_values': data[col].nunique(),
                'top_values': data[col].value_counts().head().to_dict()
            }
        summary['categorical_statistics'] = categorical_stats
    
    return summary


def log_data_info(data: Union[pd.DataFrame, np.ndarray], name: str = "Data") -> None:
    """Log information about data.
    
    Args:
        data: Data to log information about.
        name: Name for the data in logs.
    """
    if isinstance(data, pd.DataFrame):
        logger.info(f"{name} shape: {data.shape}")
        logger.info(f"{name} memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.debug(f"{name} dtypes: {data.dtypes.value_counts().to_dict()}")
    elif isinstance(data, np.ndarray):
        logger.info(f"{name} shape: {data.shape}")
        logger.info(f"{name} dtype: {data.dtype}")
        logger.info(f"{name} memory usage: {data.nbytes / 1024**2:.1f} MB")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value to return if denominator is zero.
        
    Returns:
        Division result or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_features(data: Union[np.ndarray, pd.DataFrame], 
                      method: str = "minmax",
                      feature_range: Tuple[float, float] = (0, 1)) -> Union[np.ndarray, pd.DataFrame]:
    """Normalize features using specified method.
    
    Args:
        data: Input data to normalize.
        method: Normalization method ('minmax', 'standard', 'robust').
        feature_range: Target range for min-max scaling.
        
    Returns:
        Normalized data.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
        original_index = data.index
        values = data.values
    else:
        values = data
        original_columns = None
        original_index = None
    
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_values = scaler.fit_transform(values)
    
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(normalized_values, columns=original_columns, index=original_index)
    else:
        return normalized_values


class DataValidator:
    """Comprehensive data validation utility."""
    
    # Constants
    INVALID_DATA_TYPE_MSG = "Data must be numpy array or pandas DataFrame"
    
    def __init__(self, min_samples: int = 100, min_features: int = 1):
        """Initialize DataValidator.
        
        Args:
            min_samples: Minimum number of samples required.
            min_features: Minimum number of features required.
        """
        self.min_samples = min_samples
        self.min_features = min_features
        self.logger = get_logger(__name__)
    
    def validate_shape(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Validate data shape requirements.
        
        Args:
            data: Input data to validate.
            
        Returns:
            True if shape is valid.
            
        Raises:
            ValueError: If shape requirements are not met.
        """
        if isinstance(data, np.ndarray):
            n_samples, n_features = data.shape
        elif isinstance(data, pd.DataFrame):
            n_samples, n_features = data.shape
        else:
            raise ValueError(self.INVALID_DATA_TYPE_MSG)
        
        if n_samples < self.min_samples:
            raise ValueError(f"Insufficient samples: {n_samples} < {self.min_samples}")
        
        if n_features < self.min_features:
            raise ValueError(f"Insufficient features: {n_features} < {self.min_features}")
        
        return True
    
    def validate_no_missing_values(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Validate that data has no missing values.
        
        Args:
            data: Input data to validate.
            
        Returns:
            True if no missing values.
            
        Raises:
            ValueError: If missing values are found.
        """
        if isinstance(data, np.ndarray):
            has_missing = np.isnan(data).any()
        elif isinstance(data, pd.DataFrame):
            has_missing = data.isnull().any().any()
        else:
            raise ValueError(self.INVALID_DATA_TYPE_MSG)
        
        if has_missing:
            raise ValueError("Data contains missing values")
        
        return True
    
    def validate_finite_values(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Validate that all values are finite.
        
        Args:
            data: Input data to validate.
            
        Returns:
            True if all values are finite.
            
        Raises:
            ValueError: If infinite values are found.
        """
        if isinstance(data, np.ndarray):
            has_infinite = not np.isfinite(data).all()
        elif isinstance(data, pd.DataFrame):
            has_infinite = not np.isfinite(data.select_dtypes(include=[np.number])).all().all()
        else:
            raise ValueError(self.INVALID_DATA_TYPE_MSG)
        
        if has_infinite:
            raise ValueError("Data contains infinite values")
        
        return True
    
    def validate_all(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Run all validation checks.
        
        Args:
            data: Input data to validate.
            
        Returns:
            True if all validations pass.
        """
        self.validate_shape(data)
        self.validate_no_missing_values(data)
        self.validate_finite_values(data)
        
        self.logger.info("Data validation passed all checks")
        return True
