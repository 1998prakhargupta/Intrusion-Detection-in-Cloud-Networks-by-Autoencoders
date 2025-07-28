"""Data preprocessing module for NIDS."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, List
import joblib
from pathlib import Path

from ..utils.constants import DataConstants
from ..utils.logger import get_logger


class DataPreprocessor:
    """Enhanced data preprocessor with robust feature engineering."""
    
    def __init__(self):
        """Initialize data preprocessor."""
        self.logger = get_logger(__name__)
        self.scaler = None
        self.encoding_maps = {}
        self.feature_stats = {}
        
    def preprocess_features(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Preprocess features with comprehensive handling.
        
        Args:
            data: Input DataFrame.
            feature_columns: List of feature column names.
            
        Returns:
            Preprocessed feature DataFrame.
        """
        self.logger.info("Starting comprehensive data preprocessing...")
        
        # Extract features
        features = data[feature_columns].copy()
        original_shape = features.shape
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Encode categorical features
        features = self._encode_categorical_features(features)
        
        # Convert to appropriate data types
        features = self._convert_data_types(features)
        
        # Log preprocessing results
        self._log_preprocessing_results(original_shape, features.shape)
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies.
        
        Args:
            features: Input features DataFrame.
            
        Returns:
            DataFrame with missing values handled.
        """
        missing_count = features.isnull().sum().sum()
        
        if missing_count > 0:
            self.logger.info(f"Handling {missing_count} missing values...")
            
            for col in features.columns:
                if features[col].isnull().any():
                    if features[col].dtype in ['object']:
                        # Categorical: use mode
                        mode_val = features[col].mode()
                        fill_value = mode_val[0] if not mode_val.empty else 'unknown'
                        features[col].fillna(fill_value, inplace=True)
                        self.logger.debug(f"Filled categorical column '{col}' with mode: {fill_value}")
                    else:
                        # Numerical: use mean
                        mean_val = features[col].mean()
                        features[col].fillna(mean_val, inplace=True)
                        self.logger.debug(f"Filled numerical column '{col}' with mean: {mean_val:.4f}")
        
        # Verify no missing values remain
        remaining_missing = features.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Still have {remaining_missing} missing values after preprocessing")
        
        return features
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using factorization.
        
        Args:
            features: Input features DataFrame.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        categorical_columns = features.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            self.logger.info(f"Encoding {len(categorical_columns)} categorical columns...")
            
            for col in categorical_columns:
                # Store original values for potential reverse mapping
                unique_values = features[col].unique()
                
                # Apply factorization
                encoded_values, unique_mapping = pd.factorize(features[col])
                features[col] = encoded_values
                
                # Store encoding map for future use
                self.encoding_maps[col] = {
                    'unique_mapping': unique_mapping,
                    'unique_count': len(unique_values)
                }
                
                self.logger.debug(f"Encoded column '{col}': {len(unique_values)} unique values")
        
        return features
    
    def _convert_data_types(self, features: pd.DataFrame) -> pd.DataFrame:
        """Convert all features to appropriate numeric types.
        
        Args:
            features: Input features DataFrame.
            
        Returns:
            DataFrame with converted data types.
        """
        # Convert all to float for consistency
        try:
            features = features.astype(float)
            self.logger.debug("All features converted to float type")
        except Exception as e:
            self.logger.warning(f"Could not convert all features to float: {e}")
            # Try to convert column by column
            for col in features.columns:
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except Exception as col_error:
                    self.logger.warning(f"Could not convert column '{col}' to numeric: {col_error}")
        
        return features
    
    def _log_preprocessing_results(self, original_shape: Tuple[int, int], final_shape: Tuple[int, int]) -> None:
        """Log preprocessing results summary.
        
        Args:
            original_shape: Original data shape.
            final_shape: Final processed data shape.
        """
        self.logger.info("Data preprocessing summary:")
        self.logger.info(f"  Original shape: {original_shape}")
        self.logger.info(f"  Final shape: {final_shape}")
        self.logger.info(f"  Categorical columns encoded: {len(self.encoding_maps)}")
        
        if hasattr(self, 'features'):
            memory_usage = self.features.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"  Memory usage: {memory_usage:.1f} MB")
    
    def separate_normal_anomalous(self, features: pd.DataFrame, class_info: Optional[pd.Series], 
                                 normal_identifier: str = 'normal') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Separate normal and anomalous data for training.
        
        Args:
            features: Preprocessed features DataFrame.
            class_info: Class labels Series.
            normal_identifier: Identifier for normal class.
            
        Returns:
            Tuple of (normal_data, anomalous_data).
        """
        if class_info is None:
            self.logger.info("No class information - treating all data as normal")
            return features.copy(), None
        
        # Find normal class
        unique_classes = class_info.unique()
        self.logger.info(f"Available classes: {unique_classes}")
        
        normal_class = None
        for cls in unique_classes:
            if str(cls).lower() == normal_identifier.lower():
                normal_class = cls
                break
        
        if normal_class is None:
            # Use most frequent class as normal
            normal_class = class_info.mode()[0]
            self.logger.warning(f"Normal class '{normal_identifier}' not found. Using most frequent: '{normal_class}'")
        
        self.logger.info(f"Using '{normal_class}' as normal class")
        
        # Separate data
        normal_mask = class_info == normal_class
        normal_data = features[normal_mask].copy()
        anomalous_data = features[~normal_mask].copy() if (~normal_mask).any() else None
        
        self.logger.info(f"Normal samples: {len(normal_data):,}")
        if anomalous_data is not None:
            self.logger.info(f"Anomalous samples: {len(anomalous_data):,}")
        
        return normal_data, anomalous_data
    
    def prepare_training_data(self, normal_data: pd.DataFrame, 
                            validation_ratio: float = 0.2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale training data.
        
        Args:
            normal_data: Normal data for training.
            validation_ratio: Ratio for validation split.
            random_state: Random state for reproducibility.
            
        Returns:
            Tuple of (train_scaled, val_scaled).
        """
        # Split normal data
        normal_train, normal_val = train_test_split(
            normal_data,
            test_size=validation_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        # Initialize and fit scaler on training data only
        self.scaler = StandardScaler()
        normal_train_scaled = self.scaler.fit_transform(normal_train)
        normal_val_scaled = self.scaler.transform(normal_val)
        
        # Store feature statistics
        self.feature_stats = {
            'train_samples': len(normal_train_scaled),
            'val_samples': len(normal_val_scaled),
            'features': normal_train_scaled.shape[1],
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }
        
        self.logger.info("Data preparation completed:")
        self.logger.info(f"  Training samples: {len(normal_train_scaled):,}")
        self.logger.info(f"  Validation samples: {len(normal_val_scaled):,}")
        self.logger.info(f"  Feature dimensions: {normal_train_scaled.shape[1]}")
        
        return normal_train_scaled, normal_val_scaled
    
    def scale_data(self, data: pd.DataFrame) -> np.ndarray:
        """Scale data using fitted scaler.
        
        Args:
            data: Data to scale.
            
        Returns:
            Scaled data array.
            
        Raises:
            ValueError: If scaler not fitted.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_training_data first.")
        
        return self.scaler.transform(data)
    
    def save_preprocessing_artifacts(self, output_dir: str) -> None:
        """Save preprocessing artifacts for later use.
        
        Args:
            output_dir: Directory to save artifacts.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save scaler
            if self.scaler is not None:
                scaler_path = output_path / "scaler.pkl"
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"Scaler saved to: {scaler_path}")
            
            # Save encoding maps
            if self.encoding_maps:
                encoding_path = output_path / "encoding_maps.pkl"
                joblib.dump(self.encoding_maps, encoding_path)
                self.logger.info(f"Encoding maps saved to: {encoding_path}")
            
            # Save feature statistics
            if self.feature_stats:
                stats_path = output_path / "feature_stats.pkl"
                joblib.dump(self.feature_stats, stats_path)
                self.logger.info(f"Feature statistics saved to: {stats_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save preprocessing artifacts: {e}")
            raise
    
    def load_preprocessing_artifacts(self, input_dir: str) -> None:
        """Load preprocessing artifacts.
        
        Args:
            input_dir: Directory containing artifacts.
        """
        input_path = Path(input_dir)
        
        try:
            # Load scaler
            scaler_path = input_path / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Scaler loaded from: {scaler_path}")
            
            # Load encoding maps
            encoding_path = input_path / "encoding_maps.pkl"
            if encoding_path.exists():
                self.encoding_maps = joblib.load(encoding_path)
                self.logger.info(f"Encoding maps loaded from: {encoding_path}")
            
            # Load feature statistics
            stats_path = input_path / "feature_stats.pkl"
            if stats_path.exists():
                self.feature_stats = joblib.load(stats_path)
                self.logger.info(f"Feature statistics loaded from: {stats_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load preprocessing artifacts: {e}")
            raise
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations.
        
        Returns:
            Dictionary containing preprocessing summary.
        """
        summary = {
            'scaler_fitted': self.scaler is not None,
            'categorical_encodings': len(self.encoding_maps),
            'feature_statistics': self.feature_stats.copy() if self.feature_stats else {}
        }
        
        if self.scaler is not None:
            summary['scaler_type'] = type(self.scaler).__name__
            summary['scaler_features'] = len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 'unknown'
        
        return summary
