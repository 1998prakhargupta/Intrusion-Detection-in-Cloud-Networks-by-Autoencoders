"""
Unit tests for DataPreprocessor module.

Tests the data preprocessing, feature engineering, and scaling functionality.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.preprocessor import DataPreprocessor
from utils.constants import DataConstants


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.scaler is None
        assert preprocessor.encoding_maps == {}
        assert preprocessor.feature_stats == {}
        assert hasattr(preprocessor, 'logger')
    
    @pytest.mark.unit
    def test_preprocess_features_success(self, sample_dataset, feature_columns):
        """Test successful feature preprocessing."""
        preprocessor = DataPreprocessor()
        
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        assert isinstance(processed_features, pd.DataFrame)
        assert not processed_features.empty
        assert len(processed_features.columns) > 0
        assert len(processed_features) == len(sample_dataset)
    
    @pytest.mark.unit
    def test_preprocess_features_with_missing_values(self, corrupted_data):
        """Test preprocessing handles missing values correctly."""
        preprocessor = DataPreprocessor()
        feature_cols = ['col1', 'col3']  # col2 has all NaN values
        
        processed = preprocessor.preprocess_features(corrupted_data, feature_cols)
        
        # Should not have any missing values after preprocessing
        assert not processed.isnull().any().any()
        assert len(processed) == len(corrupted_data)
    
    @pytest.mark.unit
    def test_handle_missing_values_numerical(self):
        """Test missing value handling for numerical columns."""
        preprocessor = DataPreprocessor()
        
        data_with_missing = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result = preprocessor._handle_missing_values(data_with_missing)
        
        # Missing value should be filled with mean
        assert not result['numeric_col'].isnull().any()
        expected_mean = data_with_missing['numeric_col'].mean()
        assert result['numeric_col'].iloc[2] == expected_mean
    
    @pytest.mark.unit
    def test_handle_missing_values_categorical(self):
        """Test missing value handling for categorical columns."""
        preprocessor = DataPreprocessor()
        
        data_with_missing = pd.DataFrame({
            'categorical_col': ['a', 'b', np.nan, 'a', 'b'],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result = preprocessor._handle_missing_values(data_with_missing)
        
        # Missing value should be filled with mode
        assert not result['categorical_col'].isnull().any()
        # Should be filled with mode ('a' or 'b', both appear twice)
        assert result['categorical_col'].iloc[2] in ['a', 'b']
    
    @pytest.mark.unit
    def test_handle_missing_values_all_missing_categorical(self):
        """Test handling when categorical column has all missing values."""
        preprocessor = DataPreprocessor()
        
        data_with_all_missing = pd.DataFrame({
            'categorical_col': [np.nan, np.nan, np.nan],
            'other_col': [1, 2, 3]
        })
        
        result = preprocessor._handle_missing_values(data_with_all_missing)
        
        # Should fill with 'unknown' when no mode available
        assert not result['categorical_col'].isnull().any()
        assert all(result['categorical_col'] == 'unknown')
    
    @pytest.mark.unit
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        
        data_with_categorical = pd.DataFrame({
            'categorical_col': ['tcp', 'udp', 'tcp', 'icmp', 'udp'],
            'numeric_col': [1, 2, 3, 4, 5]
        })
        
        result = preprocessor._encode_categorical_features(data_with_categorical)
        
        # Categorical column should be encoded as numeric
        assert result['categorical_col'].dtype in [np.int64, np.int32]
        # Numeric column should remain unchanged
        assert result['numeric_col'].dtype in [np.int64, np.int32, np.float64]
        # Should have encoding map stored
        assert 'categorical_col' in preprocessor.encoding_maps
    
    @pytest.mark.unit
    def test_encode_categorical_features_consistency(self):
        """Test that categorical encoding is consistent."""
        preprocessor = DataPreprocessor()
        
        data1 = pd.DataFrame({'cat_col': ['a', 'b', 'c']})
        data2 = pd.DataFrame({'cat_col': ['b', 'a', 'c', 'a']})
        
        # Encode first dataset
        result1 = preprocessor._encode_categorical_features(data1)
        
        # Encode second dataset (should use same encoding)
        result2 = preprocessor._encode_categorical_features(data2)
        
        # Same values should have same encodings
        a_encoding = result1[result1.index[0]]['cat_col']  # 'a' encoding from first dataset
        b_encoding = result1[result1.index[1]]['cat_col']  # 'b' encoding from first dataset
        
        # Check that 'a' and 'b' have consistent encodings in second dataset
        assert result2[result2.index[1]]['cat_col'] == a_encoding  # 'a' in second dataset
        assert result2[result2.index[0]]['cat_col'] == b_encoding  # 'b' in second dataset
    
    @pytest.mark.unit
    def test_convert_data_types(self):
        """Test data type conversion."""
        preprocessor = DataPreprocessor()
        
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'object_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = preprocessor._convert_data_types(mixed_data)
        
        # Should attempt to convert to appropriate numeric types
        assert result['int_col'].dtype in [np.int64, np.int32, np.float64]
        assert result['float_col'].dtype in [np.float64, np.float32]
    
    @pytest.mark.unit
    def test_separate_normal_anomalous_success(self, sample_dataset, feature_columns):
        """Test successful separation of normal and anomalous data."""
        preprocessor = DataPreprocessor()
        
        # First preprocess the features
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], normal_identifier='normal'
        )
        
        assert normal_data is not None
        assert anomalous_data is not None
        assert len(normal_data) > 0
        assert len(anomalous_data) > 0
        assert len(normal_data) + len(anomalous_data) == len(processed_features)
    
    @pytest.mark.unit
    def test_separate_normal_anomalous_only_normal(self):
        """Test separation when only normal data exists."""
        preprocessor = DataPreprocessor()
        
        features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        class_info = pd.Series(['normal', 'normal', 'normal'])
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            features, class_info, normal_identifier='normal'
        )
        
        assert normal_data is not None
        assert len(normal_data) == 3
        assert anomalous_data is None or len(anomalous_data) == 0
    
    @pytest.mark.unit
    def test_separate_normal_anomalous_only_anomalous(self):
        """Test separation when only anomalous data exists."""
        preprocessor = DataPreprocessor()
        
        features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        class_info = pd.Series(['attack', 'probe', 'dos'])
        
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            features, class_info, normal_identifier='normal'
        )
        
        assert normal_data is None or len(normal_data) == 0
        assert anomalous_data is not None
        assert len(anomalous_data) == 3
    
    @pytest.mark.unit
    def test_prepare_training_data_success(self, normal_data_sample):
        """Test successful training data preparation."""
        preprocessor = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        
        train_data, val_data = preprocessor.prepare_training_data(
            normal_df, validation_ratio=0.2, random_state=42
        )
        
        assert train_data is not None
        assert val_data is not None
        assert len(train_data) > len(val_data)  # Training should be larger
        assert len(train_data) + len(val_data) == len(normal_df)
        
        # Check that it's numpy arrays
        assert isinstance(train_data, np.ndarray)
        assert isinstance(val_data, np.ndarray)
    
    @pytest.mark.unit
    def test_prepare_training_data_different_ratios(self, normal_data_sample, validation_splits):
        """Test training data preparation with different validation ratios."""
        preprocessor = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        
        validation_ratio = validation_splits['validation_ratio']
        train_data, val_data = preprocessor.prepare_training_data(
            normal_df, validation_ratio=validation_ratio, random_state=42
        )
        
        expected_val_size = int(len(normal_df) * validation_ratio)
        expected_train_size = len(normal_df) - expected_val_size
        
        assert len(val_data) == expected_val_size
        assert len(train_data) == expected_train_size
    
    @pytest.mark.unit
    def test_prepare_training_data_reproducibility(self, normal_data_sample):
        """Test that training data preparation is reproducible with same random state."""
        preprocessor1 = DataPreprocessor()
        preprocessor2 = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        
        train1, val1 = preprocessor1.prepare_training_data(normal_df, random_state=42)
        train2, val2 = preprocessor2.prepare_training_data(normal_df, random_state=42)
        
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
    
    @pytest.mark.unit
    def test_scale_data_first_time(self, normal_data_sample):
        """Test data scaling when scaler is not fitted."""
        preprocessor = DataPreprocessor()
        data_df = pd.DataFrame(normal_data_sample)
        
        scaled_data = preprocessor.scale_data(data_df)
        
        assert isinstance(scaled_data, np.ndarray)
        assert scaled_data.shape == normal_data_sample.shape
        assert preprocessor.scaler is not None
        assert isinstance(preprocessor.scaler, StandardScaler)
        
        # Check that data is standardized (approximately mean=0, std=1)
        assert abs(np.mean(scaled_data)) < 0.1  # Close to 0
        assert abs(np.std(scaled_data) - 1.0) < 0.1  # Close to 1
    
    @pytest.mark.unit
    def test_scale_data_with_fitted_scaler(self, normal_data_sample):
        """Test data scaling when scaler is already fitted."""
        preprocessor = DataPreprocessor()
        
        # First scaling to fit the scaler
        data_df1 = pd.DataFrame(normal_data_sample)
        scaled_data1 = preprocessor.scale_data(data_df1)
        
        # Second scaling should use the same fitted scaler
        data_df2 = pd.DataFrame(normal_data_sample * 2)  # Different data
        scaled_data2 = preprocessor.scale_data(data_df2)
        
        assert isinstance(scaled_data2, np.ndarray)
        assert scaled_data2.shape == normal_data_sample.shape
        # Results should be different because data is different but scaler is same
        assert not np.array_equal(scaled_data1, scaled_data2)
    
    @pytest.mark.unit
    def test_get_preprocessing_summary(self, sample_dataset, feature_columns):
        """Test preprocessing summary generation."""
        preprocessor = DataPreprocessor()
        
        # Do some preprocessing to populate the summary
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        if normal_data is not None:
            preprocessor.scale_data(normal_data)
        
        summary = preprocessor.get_preprocessing_summary()
        
        assert isinstance(summary, dict)
        assert 'scaler_fitted' in summary
        assert 'categorical_encodings' in summary
        assert 'feature_stats' in summary
        
        if preprocessor.scaler is not None:
            assert summary['scaler_fitted'] is True
        assert summary['categorical_encodings'] == len(preprocessor.encoding_maps)
    
    @pytest.mark.unit
    def test_save_preprocessing_artifacts(self, sample_dataset, feature_columns, test_data_dir):
        """Test saving preprocessing artifacts."""
        preprocessor = DataPreprocessor()
        
        # Do preprocessing to create artifacts
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        if normal_data is not None:
            preprocessor.scale_data(normal_data)
        
        artifacts_dir = test_data_dir / "artifacts"
        preprocessor.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Check that files are created
        assert artifacts_dir.exists()
        # Should have scaler and encoding files
        scaler_file = artifacts_dir / "scaler.pkl"
        encodings_file = artifacts_dir / "encodings.pkl"
        
        if preprocessor.scaler is not None:
            assert scaler_file.exists()
        if preprocessor.encoding_maps:
            assert encodings_file.exists()
    
    @pytest.mark.unit
    def test_load_preprocessing_artifacts(self, sample_dataset, feature_columns, test_data_dir):
        """Test loading preprocessing artifacts."""
        # First create and save artifacts
        preprocessor1 = DataPreprocessor()
        processed_features = preprocessor1.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor1.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        if normal_data is not None:
            preprocessor1.scale_data(normal_data)
        
        artifacts_dir = test_data_dir / "artifacts"
        preprocessor1.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Now load artifacts in new preprocessor
        preprocessor2 = DataPreprocessor()
        preprocessor2.load_preprocessing_artifacts(str(artifacts_dir))
        
        # Check that artifacts were loaded
        if preprocessor1.scaler is not None:
            assert preprocessor2.scaler is not None
            # Test that they produce same results
            test_data = normal_data.iloc[:5] if normal_data is not None else pd.DataFrame(np.random.randn(5, 5))
            scaled1 = preprocessor1.scale_data(test_data)
            scaled2 = preprocessor2.scale_data(test_data)
            np.testing.assert_array_almost_equal(scaled1, scaled2)
    
    @pytest.mark.unit
    def test_empty_dataframe_handling(self, empty_dataframe):
        """Test handling of empty DataFrames."""
        preprocessor = DataPreprocessor()
        
        # Should handle empty DataFrame gracefully
        with pytest.raises(Exception):  # Should raise some kind of error
            preprocessor.preprocess_features(empty_dataframe, [])
    
    @pytest.mark.unit
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        preprocessor = DataPreprocessor()
        
        single_row_df = pd.DataFrame({
            'feature1': [1.0],
            'feature2': ['category_a'],
            'class': ['normal']
        })
        
        processed = preprocessor.preprocess_features(single_row_df, ['feature1', 'feature2'])
        
        assert len(processed) == 1
        assert not processed.isnull().any().any()
    
    @pytest.mark.unit
    def test_all_categorical_features(self):
        """Test preprocessing when all features are categorical."""
        preprocessor = DataPreprocessor()
        
        categorical_df = pd.DataFrame({
            'protocol': ['tcp', 'udp', 'tcp', 'icmp'],
            'service': ['http', 'ftp', 'http', 'dns'],
            'flag': ['SF', 'S0', 'SF', 'REJ'],
            'class': ['normal', 'attack', 'normal', 'normal']
        })
        
        processed = preprocessor.preprocess_features(
            categorical_df, ['protocol', 'service', 'flag']
        )
        
        # All columns should be numeric after encoding
        for col in processed.columns:
            assert processed[col].dtype in [np.int64, np.int32, np.float64]
    
    @pytest.mark.unit
    def test_all_numeric_features(self):
        """Test preprocessing when all features are numeric."""
        preprocessor = DataPreprocessor()
        
        numeric_df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'feature3': [100, 200, 300, 400],
            'class': ['normal', 'attack', 'normal', 'normal']
        })
        
        processed = preprocessor.preprocess_features(
            numeric_df, ['feature1', 'feature2', 'feature3']
        )
        
        # Should have same shape and no missing values
        assert processed.shape == (4, 3)
        assert not processed.isnull().any().any()
    
    @pytest.mark.unit
    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        preprocessor = DataPreprocessor()
        
        extreme_df = pd.DataFrame({
            'normal_feature': [1, 2, 3, 4, 5],
            'extreme_feature': [1, 2, 1000000, 4, 5],  # One very large value
            'inf_feature': [1, 2, np.inf, 4, 5],  # Infinity value
            'class': ['normal'] * 5
        })
        
        # Should handle extreme values without crashing
        processed = preprocessor.preprocess_features(
            extreme_df, ['normal_feature', 'extreme_feature', 'inf_feature']
        )
        
        assert not processed.isnull().any().any()
        # Inf should be handled somehow (replaced or removed)
        assert not np.isinf(processed['inf_feature']).any()


class TestDataPreprocessorIntegration:
    """Integration tests for DataPreprocessor with complete workflows."""
    
    @pytest.mark.integration
    def test_complete_preprocessing_workflow(self, sample_dataset, feature_columns):
        """Test complete preprocessing workflow."""
        preprocessor = DataPreprocessor()
        
        # Step 1: Preprocess features
        processed_features = preprocessor.preprocess_features(sample_dataset, feature_columns)
        assert not processed_features.empty
        
        # Step 2: Separate normal and anomalous
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        assert normal_data is not None
        
        # Step 3: Prepare training data
        train_data, val_data = preprocessor.prepare_training_data(normal_data)
        assert train_data is not None
        assert val_data is not None
        
        # Step 4: Scale anomalous data
        if anomalous_data is not None:
            scaled_anomalous = preprocessor.scale_data(anomalous_data)
            assert scaled_anomalous is not None
        
        # Step 5: Get summary
        summary = preprocessor.get_preprocessing_summary()
        assert summary['scaler_fitted'] is True
    
    @pytest.mark.integration
    def test_preprocessing_with_artifacts_save_load(self, sample_dataset, feature_columns, test_data_dir):
        """Test preprocessing with artifact saving and loading."""
        # First preprocessor - do processing and save
        preprocessor1 = DataPreprocessor()
        
        processed_features = preprocessor1.preprocess_features(sample_dataset, feature_columns)
        normal_data, _ = preprocessor1.separate_normal_anomalous(
            processed_features, sample_dataset['class'], 'normal'
        )
        train_data, val_data = preprocessor1.prepare_training_data(normal_data)
        
        artifacts_dir = test_data_dir / "artifacts"
        preprocessor1.save_preprocessing_artifacts(str(artifacts_dir))
        
        # Second preprocessor - load artifacts and use
        preprocessor2 = DataPreprocessor()
        preprocessor2.load_preprocessing_artifacts(str(artifacts_dir))
        
        # Should be able to scale new data with loaded artifacts
        new_data = pd.DataFrame(np.random.randn(10, len(feature_columns)), columns=feature_columns)
        processed_new = preprocessor2.preprocess_features(new_data, feature_columns)
        scaled_new = preprocessor2.scale_data(processed_new)
        
        assert scaled_new is not None
        assert scaled_new.shape[0] == 10
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_with_large_dataset(self, large_dataset):
        """Test preprocessing performance with large datasets."""
        import time
        
        preprocessor = DataPreprocessor()
        feature_cols = [col for col in large_dataset.columns if col != 'class']
        
        start_time = time.time()
        
        # Full preprocessing workflow
        processed_features = preprocessor.preprocess_features(large_dataset, feature_cols)
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, large_dataset['class'], 'normal'
        )
        
        if normal_data is not None:
            train_data, val_data = preprocessor.prepare_training_data(normal_data)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 30.0  # 30 seconds max
        assert not processed_features.empty
