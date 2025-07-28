"""
Unit tests for DataLoader module.

Tests the data loading, validation, and feature extraction functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.loader import DataLoader
from utils.constants import DataConstants, ValidationConstants


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        
        assert loader.data is None
        assert loader.feature_columns is None
        assert loader.class_info is None
        assert hasattr(loader, 'logger')
    
    @pytest.mark.unit
    def test_load_and_validate_data_success(self, sample_csv_file):
        """Test successful data loading and validation."""
        loader = DataLoader()
        
        data = loader.load_and_validate_data(str(sample_csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert loader.data is not None
        assert len(data) > 0
        assert 'class' in data.columns
    
    @pytest.mark.unit
    def test_load_and_validate_data_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_validate_data("non_existent_file.csv")
    
    @pytest.mark.unit
    def test_load_and_validate_data_empty_file(self, test_data_dir):
        """Test loading empty CSV file raises ValueError."""
        empty_file = test_data_dir / "empty.csv"
        empty_file.write_text("")
        
        loader = DataLoader()
        
        with pytest.raises(Exception):  # Could be ValueError or pandas error
            loader.load_and_validate_data(str(empty_file))
    
    @pytest.mark.unit
    def test_validate_data_empty_dataframe(self, empty_dataframe):
        """Test validation fails for empty DataFrame."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            loader._validate_data(empty_dataframe)
    
    @pytest.mark.unit
    def test_validate_data_insufficient_samples(self, small_dataset):
        """Test validation fails for insufficient samples."""
        loader = DataLoader()
        
        # Mock ValidationConstants to require more samples than available
        with patch.object(ValidationConstants, 'MIN_SAMPLES_REQUIRED', 1000):
            with pytest.raises(ValueError, match="Insufficient samples"):
                loader._validate_data(small_dataset)
    
    @pytest.mark.unit
    def test_validate_data_insufficient_features(self):
        """Test validation fails for insufficient features."""
        loader = DataLoader()
        minimal_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        
        # Mock ValidationConstants to require more features
        with patch.object(ValidationConstants, 'MIN_FEATURES_REQUIRED', 10):
            with pytest.raises(ValueError, match="Insufficient features"):
                loader._validate_data(minimal_df)
    
    @pytest.mark.unit 
    def test_validate_data_too_many_missing_values(self, corrupted_data):
        """Test validation fails for too many missing values."""
        loader = DataLoader()
        
        # Mock ValidationConstants to have strict missing value requirement
        with patch.object(ValidationConstants, 'MAX_MISSING_RATIO', 0.1):
            with pytest.raises(ValueError, match="Too many missing values"):
                loader._validate_data(corrupted_data)
    
    @pytest.mark.unit
    def test_extract_features_and_labels_success(self, sample_dataset):
        """Test successful feature and label extraction."""
        loader = DataLoader()
        loader.data = sample_dataset
        
        features, labels = loader.extract_features_and_labels()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'class' not in features  # Class should be excluded
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_dataset)
        assert loader.feature_columns == features
        assert loader.class_info is not None
    
    @pytest.mark.unit
    def test_extract_features_and_labels_with_data_parameter(self, sample_dataset):
        """Test feature extraction with explicit data parameter."""
        loader = DataLoader()
        
        features, labels = loader.extract_features_and_labels(sample_dataset)
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'class' not in features
        assert isinstance(labels, pd.Series)
    
    @pytest.mark.unit
    def test_extract_features_and_labels_no_data(self):
        """Test feature extraction fails when no data available."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data available"):
            loader.extract_features_and_labels()
    
    @pytest.mark.unit
    def test_extract_features_and_labels_no_class_column(self):
        """Test feature extraction when no class column exists."""
        loader = DataLoader()
        data_without_class = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        features, labels = loader.extract_features_and_labels(data_without_class)
        
        assert isinstance(features, list)
        assert len(features) == 2
        assert labels is None
    
    @pytest.mark.unit
    def test_get_data_summary_success(self, sample_dataset):
        """Test successful data summary generation."""
        loader = DataLoader()
        loader.data = sample_dataset
        
        summary = loader.get_data_summary()
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'memory_usage_mb' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'missing_ratio' in summary
        assert 'numeric_columns' in summary
        assert 'categorical_columns' in summary
        
        # Check specific values
        assert summary['shape'] == sample_dataset.shape
        assert summary['missing_values'] >= 0
        assert 0 <= summary['missing_ratio'] <= 1
    
    @pytest.mark.unit
    def test_get_data_summary_with_class_distribution(self, sample_dataset):
        """Test data summary includes class distribution when class column exists."""
        loader = DataLoader()
        
        summary = loader.get_data_summary(sample_dataset)
        
        assert 'class_distribution' in summary
        assert isinstance(summary['class_distribution'], dict)
        assert 'normal' in summary['class_distribution']
    
    @pytest.mark.unit
    def test_get_data_summary_no_data(self):
        """Test data summary fails when no data available."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data available"):
            loader.get_data_summary()
    
    @pytest.mark.unit
    def test_log_class_distribution(self, sample_dataset):
        """Test class distribution logging."""
        loader = DataLoader()
        
        # Mock logger to capture log messages
        with patch.object(loader, 'logger') as mock_logger:
            loader._log_class_distribution(sample_dataset['class'])
            
            # Verify logging calls were made
            assert mock_logger.info.called
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any('Class distribution:' in arg for arg in call_args)
    
    @pytest.mark.unit
    def test_save_processed_data_success(self, sample_dataset, test_data_dir):
        """Test successful data saving."""
        loader = DataLoader()
        output_path = test_data_dir / "saved_data.csv"
        
        loader.save_processed_data(sample_dataset, str(output_path))
        
        assert output_path.exists()
        
        # Verify saved data
        loaded_data = pd.read_csv(output_path)
        assert len(loaded_data) == len(sample_dataset)
        assert list(loaded_data.columns) == list(sample_dataset.columns)
    
    @pytest.mark.unit
    def test_save_processed_data_creates_directory(self, sample_dataset, test_data_dir):
        """Test data saving creates necessary directories."""
        loader = DataLoader()
        nested_path = test_data_dir / "nested" / "directory" / "data.csv"
        
        loader.save_processed_data(sample_dataset, str(nested_path))
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    @pytest.mark.unit
    def test_save_processed_data_permission_error(self, sample_dataset):
        """Test data saving handles permission errors."""
        loader = DataLoader()
        
        # Try to save to a path that should cause permission error
        with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                loader.save_processed_data(sample_dataset, "/root/forbidden.csv")
    
    @pytest.mark.unit
    def test_memory_usage_calculation(self, large_dataset):
        """Test memory usage calculation for large datasets."""
        loader = DataLoader()
        
        summary = loader.get_data_summary(large_dataset)
        
        assert summary['memory_usage_mb'] > 0
        assert isinstance(summary['memory_usage_mb'], float)
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_large_dataset_handling(self, large_dataset):
        """Test handling of large datasets."""
        loader = DataLoader()
        
        # This should not raise memory errors or timeout
        features, labels = loader.extract_features_and_labels(large_dataset)
        summary = loader.get_data_summary(large_dataset)
        
        assert len(features) > 0
        assert summary['shape'][0] == len(large_dataset)
    
    @pytest.mark.unit
    def test_data_types_in_summary(self, sample_dataset):
        """Test data types are correctly identified in summary."""
        loader = DataLoader()
        
        summary = loader.get_data_summary(sample_dataset)
        
        # Check that numeric and categorical columns are counted
        numeric_cols = sample_dataset.select_dtypes(include=[np.number]).columns
        categorical_cols = sample_dataset.select_dtypes(include=['object']).columns
        
        assert summary['numeric_columns'] == len(numeric_cols)
        assert summary['categorical_columns'] == len(categorical_cols)
    
    @pytest.mark.unit
    def test_missing_values_calculation(self):
        """Test missing values calculation."""
        loader = DataLoader()
        
        # Create data with known missing values
        data_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, np.nan, 3, np.nan, 5],
            'col3': [1, 2, 3, 4, 5]
        })
        
        summary = loader.get_data_summary(data_with_missing)
        
        # Should have 3 missing values total
        assert summary['missing_values'] == 3
        assert summary['missing_ratio'] == 3 / (5 * 3)  # 3 missing out of 15 total values
    
    @pytest.mark.unit
    def test_excluded_columns_are_excluded(self, sample_dataset):
        """Test that configured excluded columns are properly excluded."""
        loader = DataLoader()
        
        features, _ = loader.extract_features_and_labels(sample_dataset)
        
        excluded_columns = DataConstants.EXCLUDED_COLUMNS
        for excluded_col in excluded_columns:
            assert excluded_col not in features
    
    @pytest.mark.unit
    def test_concurrent_data_loading(self, sample_csv_file):
        """Test that multiple DataLoader instances can work concurrently."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def load_data():
            loader = DataLoader()
            try:
                data = loader.load_and_validate_data(str(sample_csv_file))
                results.put(('success', len(data)))
            except Exception as e:
                results.put(('error', str(e)))
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=load_data)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            status, result = results.get()
            if status == 'success':
                success_count += 1
                assert result > 0  # Should have loaded some data
        
        assert success_count == 3  # All threads should succeed


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with real-world scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_full_workflow_with_real_data(self, sample_csv_file, feature_columns):
        """Test complete DataLoader workflow with realistic data."""
        loader = DataLoader()
        
        # Step 1: Load and validate
        data = loader.load_and_validate_data(str(sample_csv_file))
        assert not data.empty
        
        # Step 2: Extract features
        features, labels = loader.extract_features_and_labels(data)
        assert len(features) > 0
        assert labels is not None
        
        # Step 3: Get summary
        summary = loader.get_data_summary(data)
        assert summary['missing_ratio'] < 0.5  # Should not have too many missing values
        
        # Step 4: Verify feature extraction consistency
        expected_features = [col for col in data.columns if col not in DataConstants.EXCLUDED_COLUMNS]
        assert set(features) == set(expected_features)
    
    @pytest.mark.integration
    def test_error_recovery_and_logging(self, test_data_dir):
        """Test error handling and logging throughout the workflow."""
        loader = DataLoader()
        
        # Test file not found error
        with pytest.raises(FileNotFoundError):
            loader.load_and_validate_data("nonexistent.csv")
        
        # Test empty data error  
        empty_file = test_data_dir / "empty.csv"
        empty_file.write_text("col1\n")  # Header only
        
        with pytest.raises(Exception):
            loader.load_and_validate_data(str(empty_file))
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_with_large_dataset(self, large_dataset):
        """Test DataLoader performance with large datasets."""
        import time
        
        loader = DataLoader()
        
        start_time = time.time()
        features, labels = loader.extract_features_and_labels(large_dataset)
        summary = loader.get_data_summary(large_dataset)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust as needed)
        assert end_time - start_time < 10.0  # 10 seconds max
        assert len(features) > 0
        assert summary['shape'][0] == len(large_dataset)
