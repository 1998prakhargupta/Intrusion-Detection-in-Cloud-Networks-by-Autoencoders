"""Data loading and validation module for NIDS."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

from ..utils.constants import DataConstants, ValidationConstants
from ..utils.logger import get_logger


class DataLoader:
    """Enhanced data loader with validation and basic analysis."""
    
    def __init__(self):
        """Initialize data loader."""
        self.logger = get_logger(__name__)
        self.data = None
        self.feature_columns = None
        self.class_info = None
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset with comprehensive checks.
        
        Args:
            file_path: Path to the dataset file.
            
        Returns:
            Loaded and validated DataFrame.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If data validation fails.
        """
        self.logger.info(f"Loading dataset from: {file_path}")
        
        # Check file existence
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            # Load data
            data = pd.read_csv(file_path)
            self.logger.info(f"Dataset loaded successfully - Shape: {data.shape}")
            
            # Basic validation
            self._validate_data(data)
            
            # Store data
            self.data = data
            
            # Log memory usage
            memory_mb = data.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"Memory usage: {memory_mb:.1f} MB")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate loaded data.
        
        Args:
            data: DataFrame to validate.
            
        Raises:
            ValueError: If validation fails.
        """
        # Check if empty
        if data.empty:
            raise ValueError("Dataset is empty")
        
        # Check minimum samples
        if len(data) < ValidationConstants.MIN_SAMPLES_REQUIRED:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {ValidationConstants.MIN_SAMPLES_REQUIRED}"
            )
        
        # Check minimum features
        if len(data.columns) < ValidationConstants.MIN_FEATURES_REQUIRED:
            raise ValueError(
                f"Insufficient features: {len(data.columns)} < {ValidationConstants.MIN_FEATURES_REQUIRED}"
            )
        
        # Check missing value ratio
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > ValidationConstants.MAX_MISSING_RATIO:
            raise ValueError(
                f"Too many missing values: {missing_ratio:.2%} > {ValidationConstants.MAX_MISSING_RATIO:.2%}"
            )
        
        self.logger.info("Data validation passed")
    
    def extract_features_and_labels(self, data: Optional[pd.DataFrame] = None) -> Tuple[List[str], Optional[pd.Series]]:
        """Extract feature columns and class information.
        
        Args:
            data: DataFrame to process. If None, uses stored data.
            
        Returns:
            Tuple of (feature_columns, class_info).
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        # Extract class information
        class_info = data['class'].copy() if 'class' in data.columns else None
        
        # Extract feature columns (exclude metadata columns)
        excluded_columns = DataConstants.EXCLUDED_COLUMNS
        feature_columns = [col for col in data.columns if col not in excluded_columns]
        
        self.feature_columns = feature_columns
        self.class_info = class_info
        
        self.logger.info(f"Feature extraction completed - {len(feature_columns)} features identified")
        
        # Log class distribution if available
        if class_info is not None:
            self._log_class_distribution(class_info)
        
        return feature_columns, class_info
    
    def _log_class_distribution(self, class_info: pd.Series) -> None:
        """Log class distribution statistics.
        
        Args:
            class_info: Series containing class labels.
        """
        class_counts = class_info.value_counts()
        self.logger.info("Class distribution:")
        
        for class_name, count in class_counts.items():
            percentage = (count / len(class_info)) * 100
            self.logger.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")
    
    def get_data_summary(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get comprehensive data summary.
        
        Args:
            data: DataFrame to summarize. If None, uses stored data.
            
        Returns:
            Dictionary containing data summary statistics.
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        summary = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': data.dtypes.value_counts().to_dict(),
            'missing_values': data.isnull().sum().sum(),
            'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
        }
        
        # Add class distribution if available
        if 'class' in data.columns:
            summary['class_distribution'] = data['class'].value_counts().to_dict()
        
        return summary
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str) -> None:
        """Save processed data to file.
        
        Args:
            data: DataFrame to save.
            output_path: Path to save the data.
        """
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise
