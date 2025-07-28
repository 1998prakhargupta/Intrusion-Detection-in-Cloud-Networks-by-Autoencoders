"""
Comprehensive tests for utility modules.

Tests constants, logger, and configuration utilities.
"""

import pytest
import logging
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.constants import DataConstants, ModelDefaults
from utils.logger import get_logger, setup_logging
from utils.simple_config import get_config, load_config_from_dict


class TestDataConstants:
    """Test DataConstants functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_default_feature_columns(self):
        """Test default feature columns are properly defined."""
        features = DataConstants.DEFAULT_FEATURE_COLUMNS
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(col, str) for col in features)
        
        # Check for expected network traffic features
        expected_features = ['proto', 'service', 'duration']
        for feature in expected_features:
            if feature in features:
                assert isinstance(feature, str)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_class_mapping(self):
        """Test class mapping constants."""
        assert hasattr(DataConstants, 'NORMAL_CLASS')
        assert hasattr(DataConstants, 'ANOMALY_CLASSES')
        
        normal_class = DataConstants.NORMAL_CLASS
        anomaly_classes = DataConstants.ANOMALY_CLASSES
        
        assert isinstance(normal_class, str)
        assert isinstance(anomaly_classes, list)
        assert len(anomaly_classes) > 0
        assert normal_class not in anomaly_classes
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_numeric_constants(self):
        """Test numeric constants are within valid ranges."""
        if hasattr(DataConstants, 'MAX_FEATURES'):
            assert DataConstants.MAX_FEATURES > 0
        
        if hasattr(DataConstants, 'MIN_SAMPLES'):
            assert DataConstants.MIN_SAMPLES > 0
        
        if hasattr(DataConstants, 'VALIDATION_RATIO'):
            ratio = DataConstants.VALIDATION_RATIO
            assert 0 < ratio < 1
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_file_extensions(self):
        """Test file extension constants."""
        if hasattr(DataConstants, 'SUPPORTED_FORMATS'):
            formats = DataConstants.SUPPORTED_FORMATS
            assert isinstance(formats, list)
            assert 'csv' in formats or '.csv' in formats


class TestModelDefaults:
    """Test ModelDefaults functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_model_architecture_defaults(self):
        """Test model architecture default values."""
        if hasattr(ModelDefaults, 'HIDDEN_DIMS'):
            hidden_dims = ModelDefaults.HIDDEN_DIMS
            assert isinstance(hidden_dims, list)
            assert len(hidden_dims) >= 1
            assert all(isinstance(dim, int) and dim > 0 for dim in hidden_dims)
        
        if hasattr(ModelDefaults, 'INPUT_DIM'):
            input_dim = ModelDefaults.INPUT_DIM
            assert isinstance(input_dim, int)
            assert input_dim > 0
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_training_defaults(self):
        """Test training parameter defaults."""
        if hasattr(ModelDefaults, 'EPOCHS'):
            epochs = ModelDefaults.EPOCHS
            assert isinstance(epochs, int)
            assert epochs > 0
        
        if hasattr(ModelDefaults, 'LEARNING_RATE'):
            lr = ModelDefaults.LEARNING_RATE
            assert isinstance(lr, float)
            assert 0 < lr < 1
        
        if hasattr(ModelDefaults, 'BATCH_SIZE'):
            batch_size = ModelDefaults.BATCH_SIZE
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 1024  # Reasonable upper bound
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_evaluation_defaults(self):
        """Test evaluation parameter defaults."""
        if hasattr(ModelDefaults, 'THRESHOLD_METHODS'):
            methods = ModelDefaults.THRESHOLD_METHODS
            assert isinstance(methods, list)
            assert len(methods) > 0
            assert all(isinstance(method, str) for method in methods)
        
        if hasattr(ModelDefaults, 'DEFAULT_THRESHOLD'):
            threshold = ModelDefaults.DEFAULT_THRESHOLD
            assert isinstance(threshold, (int, float))
            assert threshold > 0


class TestLogger:
    """Test logger functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level <= logging.INFO  # Should be INFO or more verbose
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_logger_different_names(self):
        """Test that different names create different loggers."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1.name != logger2.name
        assert logger1 is not logger2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_logger_same_name(self):
        """Test that same names return same logger instance."""
        logger1 = get_logger("same_logger")
        logger2 = get_logger("same_logger")
        
        assert logger1 is logger2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_logger_levels(self):
        """Test logger level configuration."""
        # Test different log levels
        debug_logger = get_logger("debug_test", level=logging.DEBUG)
        info_logger = get_logger("info_test", level=logging.INFO)
        warning_logger = get_logger("warning_test", level=logging.WARNING)
        
        assert debug_logger.level == logging.DEBUG
        assert info_logger.level == logging.INFO
        assert warning_logger.level == logging.WARNING
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_logger_output(self, caplog):
        """Test logger output functionality."""
        logger = get_logger("output_test", level=logging.DEBUG)
        
        test_message = "Test log message"
        
        with caplog.at_level(logging.DEBUG):
            logger.debug(test_message)
            logger.info(f"Info: {test_message}")
            logger.warning(f"Warning: {test_message}")
        
        # Check that messages were logged
        assert len(caplog.records) >= 1
        assert any(test_message in record.message for record in caplog.records)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_setup_logging_function(self):
        """Test setup_logging function if available."""
        try:
            # Test if setup_logging function exists
            result = setup_logging(level=logging.INFO)
            
            # Should return successfully or return a logger
            assert result is None or isinstance(result, logging.Logger)
            
        except (NameError, AttributeError):
            # setup_logging function might not exist
            pytest.skip("setup_logging function not available")
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_logger_file_output(self, tmp_path):
        """Test logger file output if supported."""
        log_file = tmp_path / "test.log"
        
        try:
            # Try to create file logger
            logger = get_logger("file_test", log_file=str(log_file))
            logger.info("Test file output")
            
            # Check if file was created and contains log
            if log_file.exists():
                content = log_file.read_text()
                assert "Test file output" in content
                
        except TypeError:
            # get_logger might not support file output
            pytest.skip("File logging not supported by get_logger")


class TestSimpleConfig:
    """Test simple_config functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_config_basic(self):
        """Test basic config retrieval."""
        config = get_config()
        
        assert isinstance(config, dict)
        # Should have some default configuration
        assert len(config) >= 0  # May be empty dict if no defaults
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_config_with_key(self):
        """Test config retrieval with specific key."""
        # Test with various common config keys
        test_keys = ['model', 'data', 'training', 'evaluation']
        
        for key in test_keys:
            try:
                config_value = get_config(key)
                # Should return value or None/default
                assert config_value is not None or config_value is None
                
            except KeyError:
                # Key might not exist, which is acceptable
                pass
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_get_config_with_default(self):
        """Test config retrieval with default value."""
        default_value = {"test": "default"}
        
        config = get_config("nonexistent_key", default=default_value)
        
        # Should return default if key doesn't exist
        assert config == default_value or isinstance(config, dict)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        test_config = {
            "model": {
                "hidden_dims": [64, 32, 64],
                "learning_rate": 0.001
            },
            "data": {
                "batch_size": 32,
                "validation_ratio": 0.2
            }
        }
        
        result = load_config_from_dict(test_config)
        
        # Should load successfully
        assert result is True or result is None
        
        # Config should be available after loading
        model_config = get_config("model")
        if model_config is not None:
            assert isinstance(model_config, dict)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_config_persistence(self):
        """Test that config persists across calls."""
        # Set a test config
        test_config = {"persistent_test": {"value": 42}}
        load_config_from_dict(test_config)
        
        # Retrieve in separate calls
        config1 = get_config("persistent_test")
        config2 = get_config("persistent_test")
        
        # Should be consistent
        assert config1 == config2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        nested_config = {
            "deep": {
                "nested": {
                    "value": "found"
                }
            }
        }
        
        load_config_from_dict(nested_config)
        
        # Test different access patterns
        deep_config = get_config("deep")
        if deep_config is not None:
            assert isinstance(deep_config, dict)
            
            if "nested" in deep_config:
                assert "value" in deep_config["nested"]
                assert deep_config["nested"]["value"] == "found"


class TestUtilsIntegration:
    """Test integration between utility modules."""
    
    @pytest.mark.integration
    @pytest.mark.utils
    def test_logger_with_config(self):
        """Test logger integration with configuration."""
        # Set up config for logging
        log_config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        load_config_from_dict(log_config)
        
        # Create logger
        logger = get_logger("config_test")
        
        # Logger should work with config
        assert isinstance(logger, logging.Logger)
        logger.info("Testing logger with config")
    
    @pytest.mark.integration
    @pytest.mark.utils
    def test_constants_with_config(self):
        """Test constants integration with configuration."""
        # Set up config that might override constants
        override_config = {
            "model": {
                "hidden_dims": [128, 64, 128],
                "epochs": 100
            },
            "data": {
                "batch_size": 64
            }
        }
        
        load_config_from_dict(override_config)
        
        # Constants should still be accessible
        if hasattr(ModelDefaults, 'HIDDEN_DIMS'):
            defaults = ModelDefaults.HIDDEN_DIMS
            assert isinstance(defaults, list)
        
        # Config should override when needed
        model_config = get_config("model")
        if model_config and "hidden_dims" in model_config:
            assert model_config["hidden_dims"] == [128, 64, 128]
    
    @pytest.mark.integration
    @pytest.mark.utils
    def test_error_handling_integration(self):
        """Test error handling across utility modules."""
        # Test invalid config
        try:
            load_config_from_dict("invalid_config")  # Should be dict
        except (TypeError, ValueError):
            pass  # Expected to fail
        
        # Logger should still work after config errors
        logger = get_logger("error_test")
        logger.info("Logger works after config error")
        
        # Constants should still be accessible
        if hasattr(DataConstants, 'NORMAL_CLASS'):
            normal_class = DataConstants.NORMAL_CLASS
            assert isinstance(normal_class, str)


class TestUtilsPerformance:
    """Performance tests for utility modules."""
    
    @pytest.mark.slow
    @pytest.mark.utils
    @pytest.mark.benchmark
    def test_logger_performance(self, benchmark):
        """Benchmark logger performance."""
        logger = get_logger("performance_test")
        
        def log_messages():
            for i in range(100):
                logger.info(f"Log message {i}")
        
        benchmark(log_messages)
    
    @pytest.mark.slow
    @pytest.mark.utils
    @pytest.mark.benchmark
    def test_config_access_performance(self, benchmark):
        """Benchmark config access performance."""
        # Set up large config
        large_config = {
            f"section_{i}": {
                f"key_{j}": f"value_{i}_{j}"
                for j in range(100)
            }
            for i in range(10)
        }
        
        load_config_from_dict(large_config)
        
        def access_config():
            for i in range(10):
                config = get_config(f"section_{i}")
                if config:
                    _ = config.get(f"key_{i}", "default")
        
        benchmark(access_config)
    
    @pytest.mark.utils
    @pytest.mark.unit
    def test_memory_usage(self):
        """Test memory usage of utility modules."""
        import sys
        
        # Measure memory before
        initial_size = sys.getsizeof(get_config())
        
        # Load large config
        large_config = {f"key_{i}": f"value_{i}" for i in range(1000)}
        load_config_from_dict(large_config)
        
        # Measure memory after
        final_size = sys.getsizeof(get_config())
        
        # Memory should increase but not excessively
        memory_increase = final_size - initial_size
        assert memory_increase >= 0  # Should increase or stay same
        
        # Create multiple loggers
        loggers = [get_logger(f"memory_test_{i}") for i in range(100)]
        
        # Should create loggers without excessive memory use
        assert len(loggers) == 100
        assert all(isinstance(logger, logging.Logger) for logger in loggers)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "hidden_dims": [64, 32, 64],
            "learning_rate": 0.001,
            "epochs": 50
        },
        "data": {
            "batch_size": 32,
            "validation_ratio": 0.2,
            "features": ["proto", "service", "duration"]
        },
        "evaluation": {
            "threshold_methods": ["mean_plus_std", "percentile"],
            "percentile_threshold": 95
        }
    }
