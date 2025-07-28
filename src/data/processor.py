"""Data processing and feature engineering for network intrusion detection."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from ..utils.logger import LoggerMixin
try:
    from ..utils.config import Config
except ImportError:
    # Fallback for when pydantic is not available
    try:
        from ..utils.config_manager import SimpleConfigManager as Config
    except ImportError:
        # Final fallback
        class Config:
            pass


class DataProcessor(LoggerMixin):
    """Data processor for network intrusion detection dataset."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize data processor.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        super().__init__()
        self.config = config or Config()
        self.scaler = None
        self.feature_names = None
        self.statistics = {}
        
        # Initialize scaler based on configuration
        self._init_scaler()
    
    def _init_scaler(self) -> None:
        """Initialize the appropriate scaler based on configuration."""
        scaling_method = self.config.data.scaling_method.lower()
        
        if scaling_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=self.config.data.feature_range)
        elif scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_method}. Using MinMax.")
            self.scaler = MinMaxScaler(feature_range=self.config.data.feature_range)
    
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from file.
        
        Args:
            data_path: Path to the dataset file.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If dataset format is not supported.
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        self.logger.info(f"Loading dataset from {data_path}")
        
        try:
            if data_path.suffix.lower() == '.csv':
                data = pd.read_csv(data_path, encoding=self.config.data.encoding)
            elif data_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(data_path)
            elif data_path.suffix.lower() in ['.json', '.jsonl']:
                data = pd.read_json(data_path, lines=(data_path.suffix == '.jsonl'))
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
            self.logger.info(f"Loaded dataset with shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset and generate statistics.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Dictionary containing dataset statistics.
        """
        self.logger.info("Analyzing dataset...")
        
        stats = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Class distribution if target column exists
        if self.config.data.target_column in data.columns:
            stats['class_distribution'] = data[self.config.data.target_column].value_counts().to_dict()
            stats['class_percentages'] = (
                data[self.config.data.target_column].value_counts(normalize=True) * 100
            ).to_dict()
        
        # Numerical column statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            stats['numerical_summary'] = data[numerical_cols].describe().to_dict()
        
        self.statistics = stats
        self.logger.info(f"Dataset analysis completed. Memory usage: {stats['memory_usage']:.2f} MB")
        
        return stats
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling missing values and outliers.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        self.logger.info("Cleaning dataset...")
        cleaned_data = data.copy()
        
        # Handle missing values
        missing_threshold = self.config.data.missing_threshold
        
        # Drop columns with too many missing values
        cols_to_drop = []
        for col in cleaned_data.columns:
            missing_pct = cleaned_data[col].isnull().sum() / len(cleaned_data)
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
                self.logger.warning(f"Dropping column {col} (missing: {missing_pct:.2%})")
        
        cleaned_data = cleaned_data.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        strategy = self.config.data.missing_strategy
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        if strategy == "mean":
            cleaned_data[numerical_cols] = cleaned_data[numerical_cols].fillna(
                cleaned_data[numerical_cols].mean()
            )
        elif strategy == "median":
            cleaned_data[numerical_cols] = cleaned_data[numerical_cols].fillna(
                cleaned_data[numerical_cols].median()
            )
        elif strategy == "mode":
            for col in cleaned_data.columns:
                mode_value = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 0
                cleaned_data[col] = cleaned_data[col].fillna(mode_value)
        elif strategy == "drop":
            cleaned_data = cleaned_data.dropna()
        
        # Handle categorical columns
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop([self.config.data.target_column], errors='ignore')
        
        for col in categorical_cols:
            mode_value = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else "unknown"
            cleaned_data[col] = cleaned_data[col].fillna(mode_value)
        
        self.logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and select features for training.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            DataFrame with selected features.
        """
        self.logger.info("Extracting features...")
        
        # Get available numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if self.config.data.target_column in numerical_cols:
            numerical_cols.remove(self.config.data.target_column)
        
        # Select features based on configuration
        selected_features = []
        for feature in self.config.data.selected_features:
            if feature in numerical_cols:
                selected_features.append(feature)
            else:
                self.logger.warning(f"Feature '{feature}' not found in dataset")
        
        # If no configured features found, use all numerical features
        if not selected_features:
            self.logger.warning("No configured features found. Using all numerical features.")
            selected_features = numerical_cols[:10]  # Limit to first 10 to avoid high dimensionality
        
        features_df = data[selected_features].copy()
        self.feature_names = selected_features
        
        self.logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return features_df
    
    def scale_features(self, 
                      features: pd.DataFrame, 
                      fit_scaler: bool = True) -> np.ndarray:
        """Scale features using the configured scaler.
        
        Args:
            features: Input features DataFrame.
            fit_scaler: Whether to fit the scaler on this data.
            
        Returns:
            Scaled features as numpy array.
        """
        self.logger.info(f"Scaling features using {type(self.scaler).__name__}")
        
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(features)
            self.logger.info("Scaler fitted and features scaled")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled_features = self.scaler.transform(features)
            self.logger.info("Features scaled using existing scaler")
        
        return scaled_features
    
    def prepare_training_data(self, 
                             data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training by separating normal and anomalous samples.
        
        Args:
            data: Input DataFrame with target column.
            
        Returns:
            Tuple of (normal_train, normal_val, anomalous_features, normal_labels).
        """
        self.logger.info("Preparing training data...")
        
        # Separate normal and anomalous data
        normal_mask = data[self.config.data.target_column] == self.config.data.normal_class
        normal_data = data[normal_mask].copy()
        anomalous_data = data[~normal_mask].copy()
        
        self.logger.info(f"Normal samples: {len(normal_data)}")
        self.logger.info(f"Anomalous samples: {len(anomalous_data)}")
        
        # Extract features for normal data
        normal_features = self.extract_features(normal_data)
        
        # Scale normal features and fit scaler
        normal_scaled = self.scale_features(normal_features, fit_scaler=True)
        
        # Split normal data into train/validation
        validation_split = self.config.training.validation_split
        normal_train, normal_val = train_test_split(
            normal_scaled,
            test_size=validation_split,
            random_state=self.config.training.seed,
            shuffle=self.config.training.shuffle
        )
        
        # Process anomalous data using the fitted scaler
        if len(anomalous_data) > 0:
            anomalous_features = self.extract_features(anomalous_data)
            anomalous_scaled = self.scale_features(anomalous_features, fit_scaler=False)
        else:
            anomalous_scaled = np.array([])
        
        # Create labels (0 = normal, 1 = anomaly)
        normal_labels = np.zeros(len(normal_val))
        
        self.logger.info(f"Training data prepared:")
        self.logger.info(f"  Normal train: {normal_train.shape}")
        self.logger.info(f"  Normal validation: {normal_val.shape}")
        self.logger.info(f"  Anomalous: {anomalous_scaled.shape}")
        
        return normal_train, normal_val, anomalous_scaled, normal_labels
    
    def save_scaler(self, output_path: Union[str, Path]) -> None:
        """Save the fitted scaler to disk.
        
        Args:
            output_path: Path to save the scaler.
        """
        if self.scaler is None:
            raise ValueError("No scaler to save. Fit scaler first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scaler_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': {
                'scaling_method': self.config.data.scaling_method,
                'feature_range': self.config.data.feature_range,
                'selected_features': self.config.data.selected_features
            }
        }
        
        joblib.dump(scaler_data, output_path)
        self.logger.info(f"Scaler saved to {output_path}")
    
    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """Load a previously saved scaler.
        
        Args:
            scaler_path: Path to the saved scaler.
        """
        scaler_path = Path(scaler_path)
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        scaler_data = joblib.load(scaler_path)
        self.scaler = scaler_data['scaler']
        self.feature_names = scaler_data['feature_names']
        
        self.logger.info(f"Scaler loaded from {scaler_path}")
        self.logger.info(f"Feature names: {self.feature_names}")
    
    def transform_new_data(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted scaler.
        
        Args:
            data: New data to transform.
            
        Returns:
            Transformed data.
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Load scaler first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Load scaler first.")
        
        # Extract features
        features = data[self.feature_names].copy()
        
        # Handle missing values (simple forward fill)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        return scaled_features
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of processed data.
        
        Returns:
            Dictionary with data processing summary.
        """
        summary = {
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'statistics': self.statistics
        }
        
        if self.scaler and hasattr(self.scaler, 'data_min_'):
            summary['feature_ranges'] = {
                'min': self.scaler.data_min_.tolist(),
                'max': self.scaler.data_max_.tolist()
            }
        
        return summary
