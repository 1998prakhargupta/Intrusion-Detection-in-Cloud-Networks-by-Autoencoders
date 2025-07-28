"""
Comprehensive unit tests for DataLoader module.

This module tests all aspects of data loading, validation, and feature extraction.
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
    """Comprehensive test suite for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        
        assert loader.data is None
        assert loader.feature_columns is None
        assert loader.class_info is None
        assert hasattr(loader, 'logger')
        assert loader.logger.name == 'data.loader'
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_success(self, sample_csv_file):
        """Test successful data loading and validation."""
        loader = DataLoader()
        
        data = loader.load_and_validate_data(str(sample_csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert loader.data is not None
        assert len(data) > 0
        assert 'class' in data.columns
        
        # Check that data is stored in loader
        pd.testing.assert_frame_equal(data, loader.data)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_validate_data("non_existent_file.csv")
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_empty_file(self, test_data_dir):
        """Test loading empty CSV file raises ValueError."""
        loader = DataLoader()
        
        # Create empty CSV file
        empty_file = test_data_dir / "empty.csv"
        pd.DataFrame().to_csv(empty_file, index=False)
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            loader.load_and_validate_data(str(empty_file))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_insufficient_samples(self, test_data_dir):
        """Test loading file with insufficient samples raises ValueError."""
        loader = DataLoader()
        
        # Create CSV with very few samples
        small_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        small_file = test_data_dir / "small.csv"
        small_data.to_csv(small_file, index=False)
        
        with patch.object(ValidationConstants, 'MIN_SAMPLES_REQUIRED', 100):
            with pytest.raises(ValueError, match="Insufficient samples"):
                loader.load_and_validate_data(str(small_file))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_insufficient_features(self, test_data_dir):
        """Test loading file with insufficient features raises ValueError."""
        loader = DataLoader()
        
        # Create CSV with very few features
        small_data = pd.DataFrame({'col1': [1, 2, 3, 4, 5] * 20})  # 100 samples, 1 feature
        small_file = test_data_dir / "few_features.csv"
        small_data.to_csv(small_file, index=False)
        
        with patch.object(ValidationConstants, 'MIN_FEATURES_REQUIRED', 5):
            with pytest.raises(ValueError, match="Insufficient features"):
                loader.load_and_validate_data(str(small_file))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_load_and_validate_data_too_many_missing_values(self, test_data_dir):
        """Test loading file with too many missing values raises ValueError."""
        loader = DataLoader()
        
        # Create CSV with many missing values
        data_with_na = pd.DataFrame({
            'col1': [1, np.nan, np.nan, np.nan, 5] * 20,
            'col2': [np.nan, 2, np.nan, np.nan, 6] * 20,
            'col3': [np.nan, np.nan, 3, np.nan, 7] * 20
        })
        na_file = test_data_dir / "many_na.csv"
        data_with_na.to_csv(na_file, index=False)
        
        with patch.object(ValidationConstants, 'MAX_MISSING_RATIO', 0.1):
            with pytest.raises(ValueError, match="Too many missing values"):
                loader.load_and_validate_data(str(na_file))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_extract_features_and_labels_success(self, sample_dataset):
        """Test successful feature and label extraction."""
        loader = DataLoader()
        loader.data = sample_dataset
        
        features, labels = loader.extract_features_and_labels()
        
        assert isinstance(features, list)
        assert isinstance(labels, pd.Series)
        assert len(features) > 0
        assert 'class' not in features
        assert len(labels) == len(sample_dataset)
        
        # Check that features are stored
        assert loader.feature_columns == features
        pd.testing.assert_series_equal(labels, loader.class_info)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_extract_features_and_labels_with_data_parameter(self, sample_dataset):
        """Test feature extraction with explicit data parameter."""
        loader = DataLoader()
        
        features, labels = loader.extract_features_and_labels(sample_dataset)
        
        assert isinstance(features, list)
        assert isinstance(labels, pd.Series)
        assert 'class' not in features
        assert all(col in sample_dataset.columns for col in features)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_extract_features_and_labels_no_data(self):
        """Test feature extraction with no data raises ValueError."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data available"):
            loader.extract_features_and_labels()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_extract_features_and_labels_no_class_column(self, test_data_dir):
        """Test feature extraction with no class column."""
        loader = DataLoader()
        
        # Create data without class column
        data_no_class = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10]
        })
        
        features, labels = loader.extract_features_and_labels(data_no_class)
        
        assert isinstance(features, list)
        assert labels is None
        assert len(features) == 2
        assert 'feature1' in features
        assert 'feature2' in features
    
    @pytest.mark.unit
    @pytest.mark.data
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
        assert 'class_distribution' in summary
        
        assert summary['shape'] == sample_dataset.shape
        assert summary['missing_values'] >= 0
        assert 0 <= summary['missing_ratio'] <= 1
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_get_data_summary_with_data_parameter(self, sample_dataset):
        """Test data summary with explicit data parameter."""
        loader = DataLoader()
        
        summary = loader.get_data_summary(sample_dataset)
        
        assert isinstance(summary, dict)
        assert summary['shape'] == sample_dataset.shape
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_get_data_summary_no_data(self):
        """Test data summary with no data raises ValueError."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data available"):
            loader.get_data_summary()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_save_processed_data(self, sample_dataset, test_data_dir):
        """Test saving processed data to file."""
        loader = DataLoader()
        output_path = test_data_dir / "processed_data.csv"
        
        loader.save_processed_data(sample_dataset, str(output_path))
        
        assert output_path.exists()
        
        # Verify saved data
        saved_data = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(saved_data, sample_dataset)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_save_processed_data_creates_directory(self, sample_dataset, test_data_dir):
        """Test saving processed data creates directory if needed."""
        loader = DataLoader()
        nested_path = test_data_dir / "nested" / "dir" / "processed_data.csv"
        
        loader.save_processed_data(sample_dataset, str(nested_path))
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    @pytest.mark.unit
    @pytest.mark.data
    @patch('data.loader.DataLoader._log_class_distribution')
    def test_log_class_distribution_called(self, mock_log_class, sample_dataset):
        """Test that class distribution logging is called when available."""
        loader = DataLoader()
        loader.data = sample_dataset
        
        features, labels = loader.extract_features_and_labels()
        
        mock_log_class.assert_called_once_with(labels)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_memory_usage_calculation(self, sample_dataset):
        """Test memory usage calculation in data summary."""
        loader = DataLoader()
        
        summary = loader.get_data_summary(sample_dataset)
        
        assert summary['memory_usage_mb'] > 0
        assert isinstance(summary['memory_usage_mb'], float)
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_full_workflow(self, sample_csv_file):
        """Test complete DataLoader workflow."""
        loader = DataLoader()
        
        # Load data
        data = loader.load_and_validate_data(str(sample_csv_file))
        
        # Extract features
        features, labels = loader.extract_features_and_labels()
        
        # Get summary
        summary = loader.get_data_summary()
        
        # Verify workflow completion
        assert data is not None
        assert features is not None
        assert labels is not None
        assert summary is not None
        assert loader.data is not None
        assert loader.feature_columns is not None
        assert loader.class_info is not None


class TestDataLoaderErrorHandling:
    """Test error handling and edge cases for DataLoader."""
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_corrupted_csv_file(self, test_data_dir):
        """Test handling of corrupted CSV files."""
        loader = DataLoader()
        
        # Create corrupted CSV file
        corrupted_file = test_data_dir / "corrupted.csv"
        with open(corrupted_file, 'w') as f:
            f.write("invalid,csv,content\n")
            f.write("missing,quotes,and\"commas\n")
            f.write("1,2\n")  # Wrong number of columns
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((pd.errors.ParserError, ValueError)):
            loader.load_and_validate_data(str(corrupted_file))
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_permission_denied(self, test_data_dir):
        """Test handling of permission denied errors."""
        loader = DataLoader()
        
        # This test may not work on all systems, so we'll skip if not applicable
        pytest.skip("Permission testing requires specific system setup")
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_very_large_dataset_warning(self, test_data_dir):
        """Test handling of very large datasets."""
        loader = DataLoader()
        
        # Create a moderately large dataset for testing
        large_data = pd.DataFrame({
            'col1': range(10000),
            'col2': range(10000),
            'class': ['normal'] * 10000
        })
        large_file = test_data_dir / "large.csv"
        large_data.to_csv(large_file, index=False)
        
        # Should load successfully but may log warnings
        data = loader.load_and_validate_data(str(large_file))
        assert len(data) == 10000


class TestDataLoaderFallbackLogic:
    """Test fallback logic and production utilities."""
    
    @pytest.mark.unit
    @pytest.mark.data
    @patch('data.loader.get_logger')
    def test_logger_fallback(self, mock_get_logger):
        """Test logger initialization fallback."""
        mock_get_logger.side_effect = ImportError("Logger not available")
        
        # Should handle logger import failure gracefully
        with patch('logging.getLogger') as mock_logging:
            mock_logging.return_value = Mock()
            loader = DataLoader()
            assert loader is not None
    
    @pytest.mark.unit
    @pytest.mark.data
    @patch('data.loader.DataConstants')
    def test_constants_fallback(self, mock_constants):
        """Test handling when constants are not available."""
        mock_constants.EXCLUDED_COLUMNS = []
        
        loader = DataLoader()
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'class': ['normal', 'attack', 'normal']
        })
        
        features, labels = loader.extract_features_and_labels(sample_data)
        
        # Should still work with fallback behavior
        assert isinstance(features, list)
        assert isinstance(labels, pd.Series)
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_validation_constants_missing(self, sample_csv_file):
        """Test behavior when validation constants are missing."""
        loader = DataLoader()
        
        with patch('data.loader.ValidationConstants', side_effect=ImportError):
            # Should use reasonable defaults or handle gracefully
            data = loader.load_and_validate_data(str(sample_csv_file))
            assert data is not None


@pytest.mark.slow
class TestDataLoaderPerformance:
    """Performance tests for DataLoader."""
    
    @pytest.mark.benchmark
    def test_load_performance(self, sample_csv_file, benchmark):
        """Benchmark data loading performance."""
        loader = DataLoader()
        
        def load_data():
            return loader.load_and_validate_data(str(sample_csv_file))
        
        result = benchmark(load_data)
        assert result is not None
    
    @pytest.mark.memory
    def test_memory_efficiency(self, test_data_dir):
        """Test memory efficiency with large datasets."""
        loader = DataLoader()
        
        # Create dataset with known memory footprint
        data_size = 1000
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(data_size) 
            for i in range(20)
        })
        large_data['class'] = ['normal'] * data_size
        
        large_file = test_data_dir / "memory_test.csv"
        large_data.to_csv(large_file, index=False)
        
        # Load and check memory usage is reasonable
        data = loader.load_and_validate_data(str(large_file))
        summary = loader.get_data_summary(data)
        
        # Memory usage should be reasonable (less than 10MB for this test)
        assert summary['memory_usage_mb'] < 10
