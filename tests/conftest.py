"""
Test configuration and shared fixtures for NIDS Autoencoder tests.

This module provides common fixtures and configuration for testing the NIDS system.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import os

# Add src to Python path for imports
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="nids_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_dataset():
    """Create a sample NIDS dataset for testing."""
    rng = np.random.default_rng(42)  # For reproducible tests
    
    # Generate normal traffic patterns
    normal_samples = 800
    normal_data = {
        'duration': rng.exponential(2.0, normal_samples),
        'protocol_type': rng.choice(['tcp', 'udp', 'icmp'], normal_samples, p=[0.7, 0.2, 0.1]),
        'service': rng.choice(['http', 'ftp', 'smtp', 'dns'], normal_samples, p=[0.5, 0.2, 0.2, 0.1]),
        'flag': rng.choice(['SF', 'S0', 'REJ'], normal_samples, p=[0.8, 0.1, 0.1]),
        'src_bytes': rng.lognormal(5, 2, normal_samples),
        'dst_bytes': rng.lognormal(4, 2, normal_samples),
        'land': rng.choice([0, 1], normal_samples, p=[0.99, 0.01]),
        'wrong_fragment': rng.poisson(0.1, normal_samples),
        'urgent': rng.poisson(0.05, normal_samples),
        'hot': rng.poisson(0.2, normal_samples),
        'num_failed_logins': rng.poisson(0.1, normal_samples),
        'class': ['normal'] * normal_samples
    }
    
    # Generate attack patterns (anomalous)
    attack_samples = 200
    attack_data = {
        'duration': rng.exponential(10.0, attack_samples),  # Longer durations
        'protocol_type': rng.choice(['tcp', 'udp', 'icmp'], attack_samples, p=[0.9, 0.05, 0.05]),
        'service': rng.choice(['http', 'ftp', 'smtp', 'dns'], attack_samples, p=[0.8, 0.1, 0.05, 0.05]),
        'flag': rng.choice(['SF', 'S0', 'REJ'], attack_samples, p=[0.3, 0.6, 0.1]),
        'src_bytes': rng.lognormal(8, 3, attack_samples),  # Larger bytes
        'dst_bytes': rng.lognormal(7, 3, attack_samples),
        'land': rng.choice([0, 1], attack_samples, p=[0.8, 0.2]),  # More land attacks
        'wrong_fragment': rng.poisson(2.0, attack_samples),  # More fragments
        'urgent': rng.poisson(1.0, attack_samples),  # More urgent
        'hot': rng.poisson(5.0, attack_samples),  # More hot indicators
        'num_failed_logins': rng.poisson(3.0, attack_samples),  # More failed logins
        'class': rng.choice(['dos', 'probe', 'r2l', 'u2r'], attack_samples)
    }
    
    # Combine normal and attack data
    combined_data = {}
    for key in normal_data.keys():
        if key == 'class':
            combined_data[key] = normal_data[key] + list(attack_data[key])
        else:
            combined_data[key] = np.concatenate([normal_data[key], attack_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Add some missing values for testing
    missing_indices = rng.choice(len(df), size=int(len(df) * 0.02), replace=False)
    missing_columns = rng.choice(['src_bytes', 'dst_bytes', 'duration'], size=len(missing_indices))
    for idx, col in zip(missing_indices, missing_columns):
        df.loc[idx, col] = np.nan
    
    return df


@pytest.fixture
def sample_csv_file(sample_dataset, test_data_dir):
    """Create a sample CSV file for testing data loading."""
    csv_path = test_data_dir / "test_dataset.csv"
    sample_dataset.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def feature_columns():
    """Return the list of feature columns (excluding class)."""
    return [
        'duration', 'protocol_type', 'service', 'flag', 
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins'
    ]


@pytest.fixture
def normal_data_sample():
    """Create a small sample of normal network traffic data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 10))  # 100 samples, 10 features
    return data


@pytest.fixture
def anomalous_data_sample():
    """Create a small sample of anomalous network traffic data."""
    np.random.seed(123)
    # Generate data with different statistical properties
    data = np.random.normal(2, 3, (20, 10))  # 20 samples, 10 features
    return data


@pytest.fixture
def trained_model_data():
    """Create mock trained model data for testing."""
    return {
        'weights': [np.random.randn(10, 5), np.random.randn(5, 3), np.random.randn(3, 5), np.random.randn(5, 10)],
        'biases': [np.random.randn(5), np.random.randn(3), np.random.randn(5), np.random.randn(10)],
        'input_dim': 10,
        'hidden_dims': [5, 3, 5],
        'is_trained': True
    }


@pytest.fixture
def mock_training_config():
    """Create mock training configuration."""
    return {
        'epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'early_stopping_patience': 5,
        'validation_split': 0.2
    }


@pytest.fixture
def mock_evaluation_results():
    """Create mock evaluation results for testing."""
    return {
        'normal_errors': {
            'mean': 0.05,
            'std': 0.02,
            'min': 0.01,
            'max': 0.15,
            'values': np.random.exponential(0.05, 100)
        },
        'anomalous_errors': {
            'mean': 0.25,
            'std': 0.10,
            'min': 0.10,
            'max': 0.80,
            'values': np.random.exponential(0.25, 20)
        },
        'thresholds': {
            'percentile': 0.12,
            'statistical': 0.09,
            'roc_optimal': 0.11
        },
        'performance': {
            'percentile': {
                'threshold': 0.12,
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.80,
                    'recall': 0.75,
                    'f1_score': 0.77
                }
            }
        },
        'roc_auc': 0.92
    }


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def corrupted_data():
    """Create corrupted data for testing error handling."""
    return pd.DataFrame({
        'col1': [1, 2, np.inf, 4, 5],
        'col2': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'class': ['normal', 'attack', 'normal', 'attack', 'normal']
    })


@pytest.fixture
def small_dataset():
    """Create a very small dataset for testing minimum requirements."""
    return pd.DataFrame({
        'feature1': [1, 2],
        'feature2': [3, 4],
        'class': ['normal', 'attack']
    })


@pytest.fixture
def large_dataset():
    """Create a larger dataset for performance testing."""
    np.random.seed(42)
    size = 10000
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, size),
        'feature2': np.random.normal(0, 1, size),
        'feature3': np.random.uniform(0, 1, size),
        'feature4': np.random.exponential(1, size),
        'feature5': np.random.poisson(2, size),
        'class': np.random.choice(['normal', 'attack'], size, p=[0.8, 0.2])
    })


@pytest.fixture
def temp_model_file(test_data_dir):
    """Create a temporary file path for model saving/loading tests."""
    return test_data_dir / "test_model.pkl"


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    import logging
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger


# Parameterized fixtures for different test scenarios
@pytest.fixture(params=[
    {'epochs': 5, 'learning_rate': 0.01},
    {'epochs': 10, 'learning_rate': 0.001},
    {'epochs': 20, 'learning_rate': 0.0001}
])
def training_configs(request):
    """Parametrized training configurations for comprehensive testing."""
    return request.param


@pytest.fixture(params=[
    'percentile',
    'statistical', 
    'roc_optimal'
])
def threshold_methods(request):
    """Parametrized threshold methods for testing."""
    return request.param


@pytest.fixture(params=[
    {'validation_ratio': 0.1},
    {'validation_ratio': 0.2},
    {'validation_ratio': 0.3}
])
def validation_splits(request):
    """Parametrized validation split ratios for testing."""
    return request.param


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require test data"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests based on name patterns
        if any(keyword in item.name for keyword in ["large", "performance", "benchmark"]):
            item.add_marker(pytest.mark.slow)


# Additional fixtures for comprehensive testing
@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        'input_dim': 20,
        'hidden_dims': [16, 8, 16],
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32,
        'early_stopping_patience': 3
    }


@pytest.fixture
def mock_training_data():
    """Mock training data for model testing."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 20
    
    # Generate normal training data
    data = rng.normal(0, 1, (n_samples, n_features))
    return data.astype(np.float32)


@pytest.fixture
def mock_evaluation_data():
    """Mock evaluation data with normal and anomalous samples."""
    rng = np.random.default_rng(42)
    n_normal = 50
    n_anomalous = 20
    n_features = 20
    
    # Normal data (similar to training)
    normal_data = rng.normal(0, 1, (n_normal, n_features))
    
    # Anomalous data (different distribution)
    anomalous_data = rng.normal(3, 2, (n_anomalous, n_features))
    
    return {
        'normal': normal_data.astype(np.float32),
        'anomalous': anomalous_data.astype(np.float32)
    }


@pytest.fixture
def temp_model_path(test_data_dir):
    """Temporary path for saving/loading models."""
    return test_data_dir / "test_model.pkl"


@pytest.fixture
def temp_config_path(test_data_dir):
    """Temporary path for configuration files."""
    return test_data_dir / "test_config.yaml"


@pytest.fixture
def sample_feature_names():
    """Sample feature names for testing."""
    return [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins'
    ]


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    rng = np.random.default_rng(42)
    n_samples = 10000
    n_features = 50
    
    # Generate large dataset
    data = rng.normal(0, 1, (n_samples, n_features))
    labels = rng.choice(['normal', 'attack'], n_samples, p=[0.8, 0.2])
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df['class'] = labels
    
    return df


@pytest.fixture
def corrupted_data_sample(test_data_dir):
    """Sample with corrupted/invalid data for error testing."""
    corrupted_csv = test_data_dir / "corrupted.csv"
    
    with open(corrupted_csv, 'w') as f:
        f.write("feature1,feature2,class\n")
        f.write("1.0,2.0,normal\n")
        f.write("invalid,data,here\n")  # Invalid row
        f.write("3.0,4.0,attack\n")
        f.write("5.0,,normal\n")  # Missing value
    
    return corrupted_csv


@pytest.fixture
def memory_efficient_config():
    """Configuration optimized for memory-constrained testing."""
    return {
        'model': {
            'hidden_dims': [8, 4, 8],  # Smaller architecture
            'batch_size': 16,
            'epochs': 5
        },
        'data': {
            'max_samples': 1000,
            'validation_ratio': 0.2
        }
    }


# Cleanup hooks
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment before and after each test."""
    # Setup
    original_env = os.environ.copy()
    
    yield
    
    # Cleanup
    os.environ.clear()
    os.environ.update(original_env)


# Skip markers for conditional tests
skip_if_no_gpu = pytest.mark.skipif(
    not os.environ.get('GPU_AVAILABLE', False),
    reason="GPU not available"
)

skip_if_no_network = pytest.mark.skipif(
    not os.environ.get('NETWORK_TESTS', False),
    reason="Network tests disabled"
)

skip_if_no_docker = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker not available"
)
