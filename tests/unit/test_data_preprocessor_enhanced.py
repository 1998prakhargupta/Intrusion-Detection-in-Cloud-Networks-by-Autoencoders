"""
Comprehensive unit tests for DataPreprocessor module.

This module tests all aspects of data preprocessing, feature engineering, and scaling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.preprocessor import DataPreprocessor
from utils.constants import DataConstants


class TestDataPreprocessor:
    """Comprehensive test suite for DataPreprocessor class."""
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.scaler is None
        assert preprocessor.encoding_maps == {}
        assert preprocessor.feature_stats == {}
        assert hasattr(preprocessor, 'logger')
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_preprocess_features_success(self, sample_dataset, feature_columns):
        """Test successful feature preprocessing."""
        preprocessor = DataPreprocessor()
        
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        assert isinstance(processed_features, pd.DataFrame)
        assert len(processed_features) == len(sample_dataset)
        assert len(processed_features.columns) <= len(feature_columns)
        
        # Check no missing values remain
        assert processed_features.isnull().sum().sum() == 0
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_handle_missing_values_numerical(self):
        """Test handling missing values in numerical columns."""
        preprocessor = DataPreprocessor()
        
        # Create data with missing numerical values
        data_with_na = pd.DataFrame({
            'num_col1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'num_col2': [10.0, np.nan, 30.0, np.nan, 50.0]
        })
        
        result = preprocessor._handle_missing_values(data_with_na)
        
        assert result.isnull().sum().sum() == 0
        
        # Check that missing values were filled with mean
        assert result.loc[2, 'num_col1'] == data_with_na['num_col1'].mean()
        assert result.loc[1, 'num_col2'] == data_with_na['num_col2'].mean()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns."""
        preprocessor = DataPreprocessor()
        
        # Create data with missing categorical values
        data_with_na = pd.DataFrame({
            'cat_col1': ['a', 'b', None, 'a', 'b'],
            'cat_col2': ['x', None, 'y', None, 'x']
        })
        
        result = preprocessor._handle_missing_values(data_with_na)
        
        assert result.isnull().sum().sum() == 0
        
        # Check that missing values were filled with mode or 'unknown'
        assert result.loc[2, 'cat_col1'] in ['a', 'b', 'unknown']
        assert result.loc[1, 'cat_col2'] in ['x', 'y', 'unknown']
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        
        # Create data with categorical features
        categorical_data = pd.DataFrame({
            'protocol': ['tcp', 'udp', 'tcp', 'icmp', 'tcp'],
            'service': ['http', 'ftp', 'http', 'dns', 'http'],
            'numeric_col': [1, 2, 3, 4, 5]
        })
        
        result = preprocessor._encode_categorical_features(categorical_data)
        
        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(result['protocol'])
        assert pd.api.types.is_numeric_dtype(result['service'])
        assert pd.api.types.is_numeric_dtype(result['numeric_col'])
        
        # Check encoding maps are populated
        assert 'protocol' in preprocessor.encoding_maps
        assert 'service' in preprocessor.encoding_maps
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_convert_data_types(self):
        """Test data type conversion."""
        preprocessor = DataPreprocessor()
        
        # Create data with mixed types
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['1', '2', '3', '4', '5']
        })
        
        result = preprocessor._convert_data_types(mixed_data)
        
        # Should attempt to convert string numbers to numeric
        assert pd.api.types.is_numeric_dtype(result['int_col'])
        assert pd.api.types.is_numeric_dtype(result['float_col'])
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_separate_normal_anomalous(self, sample_dataset):
        """Test separation of normal and anomalous data."""
        preprocessor = DataPreprocessor()
        
        # Preprocess features first
        feature_columns = [col for col in sample_dataset.columns if col != 'class']
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], normal_identifier='normal'
        )
        
        assert normal_data is not None
        assert anomalous_data is not None
        assert isinstance(normal_data, pd.DataFrame)
        assert isinstance(anomalous_data, pd.DataFrame)
        assert len(normal_data) + len(anomalous_data) == len(processed_features)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_separate_normal_anomalous_no_anomalies(self):
        """Test separation when no anomalies present."""
        preprocessor = DataPreprocessor()
        
        # Create data with only normal samples
        features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        class_info = pd.Series(['normal', 'normal', 'normal'])
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            features, class_info, normal_identifier='normal'
        )
        
        assert normal_data is not None
        assert anomalous_data is None
        assert len(normal_data) == 3
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_prepare_training_data(self, normal_data_sample):
        """Test training data preparation."""
        preprocessor = DataPreprocessor()
        
        # Convert numpy array to DataFrame
        normal_df = pd.DataFrame(normal_data_sample)
        
        train_data, val_data = preprocessor.prepare_training_data(
            normal_df, validation_ratio=0.2, random_state=42
        )
        
        assert isinstance(train_data, np.ndarray)
        assert isinstance(val_data, np.ndarray)
        assert len(train_data) + len(val_data) == len(normal_df)
        assert len(val_data) / len(normal_df) == pytest.approx(0.2, abs=0.05)
        
        # Check that scaler was fitted
        assert preprocessor.scaler is not None
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_prepare_training_data_different_ratios(self, normal_data_sample):
        """Test training data preparation with different validation ratios."""
        preprocessor = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        
        for ratio in [0.1, 0.3, 0.5]:
            train_data, val_data = preprocessor.prepare_training_data(
                normal_df, validation_ratio=ratio, random_state=42
            )
            
            expected_val_size = int(len(normal_df) * ratio)
            assert abs(len(val_data) - expected_val_size) <= 1  # Allow for rounding
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_scale_data(self, normal_data_sample):
        """Test data scaling functionality."""
        preprocessor = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        
        # First fit the scaler
        _, _ = preprocessor.prepare_training_data(normal_df, validation_ratio=0.2)
        
        # Now test scaling new data
        new_data = pd.DataFrame(np.random.randn(10, normal_data_sample.shape[1]))
        scaled_data = preprocessor.scale_data(new_data)
        
        assert isinstance(scaled_data, np.ndarray)
        assert scaled_data.shape == new_data.shape
        
        # Scaled data should have similar range to training data
        assert np.abs(scaled_data.mean()) < 2  # Should be roughly centered
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_scale_data_without_fitted_scaler(self):
        """Test scaling data without fitted scaler raises error."""
        preprocessor = DataPreprocessor()
        
        data = pd.DataFrame(np.random.randn(10, 5))
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.scale_data(data)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_get_preprocessing_summary(self, sample_dataset, feature_columns):
        """Test preprocessing summary generation."""
        preprocessor = DataPreprocessor()
        
        # Run preprocessing
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        preprocessor.prepare_training_data(normal_data, validation_ratio=0.2)
        
        summary = preprocessor.get_preprocessing_summary()
        
        assert isinstance(summary, dict)
        assert 'scaler_fitted' in summary
        assert 'categorical_encodings' in summary
        assert summary['scaler_fitted'] is True
        assert isinstance(summary['categorical_encodings'], int)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_save_preprocessing_artifacts(self, sample_dataset, feature_columns, test_data_dir):
        """Test saving preprocessing artifacts."""
        preprocessor = DataPreprocessor()
        
        # Run preprocessing to generate artifacts
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        preprocessor.prepare_training_data(normal_data, validation_ratio=0.2)
        
        artifacts_dir = test_data_dir / "artifacts"
        preprocessor.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Check that artifacts were saved
        assert (artifacts_dir / "scaler.pkl").exists()
        assert (artifacts_dir / "encoding_maps.pkl").exists()
        assert (artifacts_dir / "feature_stats.pkl").exists()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_preprocessing_artifacts(self, sample_dataset, feature_columns, test_data_dir):
        """Test loading preprocessing artifacts."""
        preprocessor1 = DataPreprocessor()
        preprocessor2 = DataPreprocessor()
        
        # Create and save artifacts with first preprocessor
        processed_features = preprocessor1.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor1.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        preprocessor1.prepare_training_data(normal_data, validation_ratio=0.2)
        
        artifacts_dir = test_data_dir / "artifacts"
        preprocessor1.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Load artifacts with second preprocessor
        preprocessor2.load_preprocessing_artifacts(str(artifacts_dir))
        
        # Check that artifacts were loaded
        assert preprocessor2.scaler is not None
        assert len(preprocessor2.encoding_maps) == len(preprocessor1.encoding_maps)


class TestDataPreprocessorErrorHandling:
    """Test error handling and edge cases for DataPreprocessor."""
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_preprocess_empty_dataframe(self):
        """Test preprocessing empty DataFrame."""
        preprocessor = DataPreprocessor()
        
        empty_df = pd.DataFrame()
        empty_features = []
        
        with pytest.raises(ValueError):
            preprocessor.preprocess_features(empty_df, empty_features)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_preprocess_invalid_feature_columns(self, sample_dataset):
        """Test preprocessing with invalid feature columns."""
        preprocessor = DataPreprocessor()
        
        invalid_features = ['non_existent_column']
        
        with pytest.raises(KeyError):
            preprocessor.preprocess_features(sample_dataset, invalid_features)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_separate_with_invalid_class_info(self, sample_dataset, feature_columns):
        """Test separation with invalid class information."""
        preprocessor = DataPreprocessor()
        
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        invalid_class_info = pd.Series(['invalid'] * len(processed_features))
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, invalid_class_info, 'normal'
        )
        
        # Should handle gracefully - no normal data found
        assert normal_data is None or len(normal_data) == 0
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_prepare_training_data_insufficient_samples(self):
        """Test training data preparation with insufficient samples."""
        preprocessor = DataPreprocessor()
        
        # Very small dataset
        small_data = pd.DataFrame([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            preprocessor.prepare_training_data(small_data, validation_ratio=0.5)


class TestDataPreprocessorFallbackLogic:
    """Test fallback logic and production utilities."""
    
    @pytest.mark.unit
    @pytest.mark.data
    @patch('data.preprocessor.get_logger')
    def test_logger_fallback(self, mock_get_logger):
        """Test logger initialization fallback."""
        mock_get_logger.side_effect = ImportError("Logger not available")
        
        with patch('logging.getLogger') as mock_logging:
            mock_logging.return_value = Mock()
            preprocessor = DataPreprocessor()
            assert preprocessor is not None
    
    @pytest.mark.unit
    @pytest.mark.data
    @patch('data.preprocessor.StandardScaler')
    def test_scaler_fallback(self, mock_scaler):
        """Test scaler initialization fallback."""
        mock_scaler.side_effect = ImportError("StandardScaler not available")
        
        preprocessor = DataPreprocessor()
        
        # Should handle gracefully or use alternative scaling
        with pytest.raises(ImportError):
            preprocessor.prepare_training_data(pd.DataFrame([[1, 2], [3, 4]]))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_missing_value_strategy_fallback(self):
        """Test fallback when missing value strategy is not available."""
        preprocessor = DataPreprocessor()
        
        with patch.object(DataConstants, 'MISSING_VALUE_STRATEGY', 'unknown_strategy'):
            # Should fall back to default behavior
            data_with_na = pd.DataFrame({
                'col1': [1, 2, np.nan, 4, 5]
            })
            
            result = preprocessor._handle_missing_values(data_with_na)
            assert result.isnull().sum().sum() == 0


class TestDataPreprocessorIntegration:
    """Integration tests for DataPreprocessor."""
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_full_preprocessing_workflow(self, sample_dataset):
        """Test complete preprocessing workflow."""
        preprocessor = DataPreprocessor()
        
        # Extract feature columns
        feature_columns = [col for col in sample_dataset.columns if col != 'class']
        
        # 1. Preprocess features
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        # 2. Separate normal/anomalous
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        
        # 3. Prepare training data
        train_data, val_data = preprocessor.prepare_training_data(
            normal_data, validation_ratio=0.2, random_state=42
        )
        
        # 4. Scale anomalous data
        if anomalous_data is not None:
            anomalous_scaled = preprocessor.scale_data(anomalous_data)
            assert anomalous_scaled is not None
            assert anomalous_scaled.shape[1] == train_data.shape[1]
        
        # Verify workflow completion
        assert processed_features is not None
        assert normal_data is not None
        assert train_data is not None
        assert val_data is not None
        assert preprocessor.scaler is not None
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_preprocessing_reproducibility(self, sample_dataset):
        """Test that preprocessing is reproducible with same random seed."""
        feature_columns = [col for col in sample_dataset.columns if col != 'class']
        
        # Run preprocessing twice with same seed
        preprocessor1 = DataPreprocessor()
        preprocessor2 = DataPreprocessor()
        
        # First run
        processed1 = preprocessor1.preprocess_features(sample_dataset, feature_columns)
        normal1, _ = preprocessor1.separate_normal_anomalous(processed1, sample_dataset['class'], 'normal')
        train1, val1 = preprocessor1.prepare_training_data(normal1, validation_ratio=0.2, random_state=42)
        
        # Second run
        processed2 = preprocessor2.preprocess_features(sample_dataset, feature_columns)
        normal2, _ = preprocessor2.separate_normal_anomalous(processed2, sample_dataset['class'], 'normal')
        train2, val2 = preprocessor2.prepare_training_data(normal2, validation_ratio=0.2, random_state=42)
        
        # Results should be identical
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)


@pytest.mark.slow
class TestDataPreprocessorPerformance:
    """Performance tests for DataPreprocessor."""
    
    @pytest.mark.benchmark
    def test_preprocessing_performance(self, sample_dataset, benchmark):
        """Benchmark preprocessing performance."""
        preprocessor = DataPreprocessor()
        feature_columns = [col for col in sample_dataset.columns if col != 'class']
        
        def run_preprocessing():
            return preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        result = benchmark(run_preprocessing)
        assert result is not None
    
    @pytest.mark.memory
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        preprocessor = DataPreprocessor()
        
        # Create large dataset
        n_samples = 10000
        n_features = 50
        
        large_data = pd.DataFrame(np.random.randn(n_samples, n_features))
        large_data.columns = [f'feature_{i}' for i in range(n_features)]
        feature_columns = list(large_data.columns)
        
        # Should handle large dataset without excessive memory usage
        processed = preprocessor.preprocess_features(large_data, feature_columns)
        assert processed is not None
        assert len(processed) == n_samples
