"""
Comprehensive unit tests for EnhancedModelTrainer module.

This module tests all aspects of model training, architecture, and management.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
from utils.constants import ModelDefaults


class TestProductionAutoencoder:
    """Test suite for ProductionAutoencoder class."""
    
    def test_initialization(self):
        """Test ProductionAutoencoder initialization."""
        input_dim = 10
        hidden_dims = [8, 6, 4, 6, 8]
        
        autoencoder = ProductionAutoencoder(input_dim, hidden_dims)
        
        assert autoencoder.input_dim == input_dim
        assert autoencoder.hidden_dims == hidden_dims
        assert len(autoencoder.weights) == 5  # 4 transitions + output
        assert len(autoencoder.biases) == 5
        assert autoencoder.is_trained is False
        assert hasattr(autoencoder, 'logger')
        assert autoencoder.training_history == []
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_initialization_default_hidden_dims(self):
        """Test initialization with default hidden dimensions."""
        input_dim = 10
        
        with patch.object(ModelDefaults, 'HIDDEN_DIMS', [8, 4, 8]):
            autoencoder = ProductionAutoencoder(input_dim)
            assert autoencoder.hidden_dims == [8, 4, 8]
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_weight_initialization(self):
        """Test proper weight initialization."""
        input_dim = 10
        hidden_dims = [8, 6, 8]
        
        autoencoder = ProductionAutoencoder(input_dim, hidden_dims)
        
        # Check weight shapes
        expected_shapes = [(10, 8), (8, 6), (6, 8), (8, 10)]
        for i, (weight, expected_shape) in enumerate(zip(autoencoder.weights, expected_shapes)):
            assert weight.shape == expected_shape, f"Weight {i} shape mismatch"
        
        # Check bias shapes
        expected_bias_shapes = [8, 6, 8, 10]
        for i, (bias, expected_size) in enumerate(zip(autoencoder.biases, expected_bias_shapes)):
            assert bias.shape == (expected_size,), f"Bias {i} shape mismatch"
        
        # Check Xavier initialization (weights should be in reasonable range)
        for weight in autoencoder.weights:
            assert np.abs(weight).max() < 2.0  # Xavier should keep weights reasonable
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_forward_pass(self, normal_data_sample):
        """Test forward pass through autoencoder."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        # Test single sample
        single_sample = normal_data_sample[0:1]
        reconstruction = autoencoder._forward(single_sample)
        
        assert reconstruction.shape == single_sample.shape
        assert np.isfinite(reconstruction).all()
        
        # Test batch
        batch_reconstruction = autoencoder._forward(normal_data_sample)
        assert batch_reconstruction.shape == normal_data_sample.shape
        assert np.isfinite(batch_reconstruction).all()
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_predict(self, normal_data_sample):
        """Test prediction method."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        predictions = autoencoder.predict(normal_data_sample)
        
        assert predictions.shape == normal_data_sample.shape
        assert np.isfinite(predictions).all()
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_reconstruction_error(self, normal_data_sample):
        """Test reconstruction error calculation."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        errors = autoencoder.reconstruction_error(normal_data_sample)
        
        assert errors.shape == (normal_data_sample.shape[0],)
        assert np.all(errors >= 0)  # Reconstruction errors should be non-negative
        assert np.isfinite(errors).all()
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_basic(self, normal_data_sample):
        """Test basic training functionality."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        # Quick training
        training_info = autoencoder.train(
            normal_data_sample, 
            epochs=5, 
            learning_rate=0.01,
            batch_size=20
        )
        
        assert autoencoder.is_trained is True
        assert len(autoencoder.training_history) > 0
        assert 'total_time' in training_info
        assert 'final_loss' in training_info
        assert 'epochs' in training_info
        assert training_info['epochs'] == 5
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_with_validation(self, normal_data_sample):
        """Test training with validation data."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        # Split data for training and validation
        split_idx = len(normal_data_sample) // 2
        train_data = normal_data_sample[:split_idx]
        val_data = normal_data_sample[split_idx:]
        
        training_info = autoencoder.train(
            train_data,
            epochs=5,
            learning_rate=0.01,
            validation_data=val_data
        )
        
        assert 'validation_loss' in training_info
        assert len(autoencoder.training_history) > 0
        
        # Check that validation loss was tracked
        history_entry = autoencoder.training_history[-1]
        assert 'val_loss' in history_entry
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_early_stopping(self, normal_data_sample):
        """Test early stopping functionality."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        # Split data
        split_idx = len(normal_data_sample) // 2
        train_data = normal_data_sample[:split_idx]
        val_data = normal_data_sample[split_idx:]
        
        training_info = autoencoder.train(
            train_data,
            epochs=100,  # High number
            learning_rate=0.01,
            validation_data=val_data,
            early_stopping_patience=3
        )
        
        # Should stop early if validation loss doesn't improve
        assert training_info['epochs'] <= 100
        assert autoencoder.is_trained is True
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_save_and_load(self, normal_data_sample, test_data_dir):
        """Test model saving and loading."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        # Train model briefly
        autoencoder.train(normal_data_sample, epochs=3)
        
        # Save model
        model_path = test_data_dir / "test_autoencoder.pkl"
        autoencoder.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_autoencoder = ProductionAutoencoder.load(str(model_path))
        
        assert loaded_autoencoder.input_dim == autoencoder.input_dim
        assert loaded_autoencoder.hidden_dims == autoencoder.hidden_dims
        assert loaded_autoencoder.is_trained == autoencoder.is_trained
        
        # Test that loaded model produces same results
        original_pred = autoencoder.predict(normal_data_sample[:5])
        loaded_pred = loaded_autoencoder.predict(normal_data_sample[:5])
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=6)


class TestEnhancedModelTrainer:
    """Test suite for EnhancedModelTrainer class."""
    
    def test_initialization(self):
        """Test EnhancedModelTrainer initialization."""
        trainer = EnhancedModelTrainer()
        
        assert hasattr(trainer, 'logger')
        assert trainer.model is None
        assert trainer.training_history is None
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_create_model(self):
        """Test model creation."""
        trainer = EnhancedModelTrainer()
        
        input_dim = 10
        hidden_dims = [8, 6, 8]
        
        model = trainer.create_model(input_dim, hidden_dims)
        
        assert isinstance(model, ProductionAutoencoder)
        assert model.input_dim == input_dim
        assert model.hidden_dims == hidden_dims
        assert trainer.model is model
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_create_model_default_dims(self):
        """Test model creation with default hidden dimensions."""
        trainer = EnhancedModelTrainer()
        
        with patch.object(ModelDefaults, 'HIDDEN_DIMS', [16, 8, 16]):
            model = trainer.create_model(input_dim=10)
            assert model.hidden_dims == [16, 8, 16]
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_model(self, normal_data_sample, mock_training_config):
        """Test model training through trainer."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [8, 6, 8])
        
        # Split data for validation
        split_idx = len(normal_data_sample) // 2
        train_data = normal_data_sample[:split_idx]
        val_data = normal_data_sample[split_idx:]
        
        # Modify config for quick test
        config = mock_training_config.copy()
        config['epochs'] = 5
        
        training_history = trainer.train_model(
            train_data=train_data,
            val_data=val_data,
            config=config
        )
        
        assert isinstance(training_history, dict)
        assert 'final_train_loss' in training_history
        assert 'final_val_loss' in training_history
        assert 'total_time' in training_history
        assert 'epochs' in training_history
        assert training_history['epochs'] == 5
        assert model.is_trained is True
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_model_no_validation(self, normal_data_sample, mock_training_config):
        """Test model training without validation data."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [8, 6, 8])
        
        config = mock_training_config.copy()
        config['epochs'] = 3
        
        training_history = trainer.train_model(
            train_data=normal_data_sample,
            val_data=None,
            config=config
        )
        
        assert 'final_train_loss' in training_history
        assert 'final_val_loss' not in training_history
        assert model.is_trained is True
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_model_invalid_config(self, normal_data_sample):
        """Test model training with invalid configuration."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [8, 6, 8])
        
        # Invalid config - negative epochs
        invalid_config = {'epochs': -1, 'learning_rate': 0.01}
        
        with pytest.raises(ValueError):
            trainer.train_model(
                train_data=normal_data_sample,
                config=invalid_config
            )
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model(self, normal_data_sample):
        """Test model evaluation."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [8, 6, 8])
        
        # Train briefly
        model.train(normal_data_sample, epochs=3)
        
        # Evaluate
        metrics = trainer.evaluate_model(normal_data_sample)
        
        assert isinstance(metrics, dict)
        assert 'mean_reconstruction_error' in metrics
        assert 'std_reconstruction_error' in metrics
        assert 'min_reconstruction_error' in metrics
        assert 'max_reconstruction_error' in metrics
        
        # Check metric values are reasonable
        assert metrics['mean_reconstruction_error'] >= 0
        assert metrics['std_reconstruction_error'] >= 0
        assert metrics['min_reconstruction_error'] >= 0
        assert metrics['max_reconstruction_error'] >= metrics['min_reconstruction_error']
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_save_model(self, normal_data_sample, test_data_dir):
        """Test saving trained model."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=3)
        
        model_path = test_data_dir / "trainer_model.pkl"
        trainer.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load and verify
        loaded_model = ProductionAutoencoder.load(str(model_path))
        assert loaded_model.is_trained is True
        assert loaded_model.input_dim == input_dim
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_load_model(self, normal_data_sample, test_data_dir):
        """Test loading model into trainer."""
        # Create and save a model first
        trainer1 = EnhancedModelTrainer()
        input_dim = normal_data_sample.shape[1]
        model1 = trainer1.create_model(input_dim, [8, 6, 8])
        model1.train(normal_data_sample, epochs=3)
        
        model_path = test_data_dir / "load_test_model.pkl"
        trainer1.save_model(str(model_path))
        
        # Load into new trainer
        trainer2 = EnhancedModelTrainer()
        trainer2.load_model(str(model_path))
        
        assert trainer2.model is not None
        assert trainer2.model.is_trained is True
        assert trainer2.model.input_dim == input_dim


class TestEnhancedModelTrainerErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_model_without_model(self, normal_data_sample):
        """Test training without creating model first."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model created"):
            trainer.train_model(normal_data_sample, config={'epochs': 5})
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model_without_model(self, normal_data_sample):
        """Test evaluation without model."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model available"):
            trainer.evaluate_model(normal_data_sample)
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_untrained_model(self, normal_data_sample):
        """Test evaluation of untrained model."""
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(normal_data_sample.shape[1], [8, 6, 8])
        
        with pytest.raises(ValueError, match="Model not trained"):
            trainer.evaluate_model(normal_data_sample)
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_save_model_without_model(self, test_data_dir):
        """Test saving without model."""
        trainer = EnhancedModelTrainer()
        model_path = test_data_dir / "no_model.pkl"
        
        with pytest.raises(ValueError, match="No model to save"):
            trainer.save_model(str(model_path))
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_model.pkl")
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_train_with_invalid_data_shape(self):
        """Test training with incompatible data shape."""
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(input_dim=10, hidden_dims=[8, 6, 8])
        
        # Wrong number of features
        wrong_data = np.random.randn(100, 5)  # 5 features instead of 10
        
        with pytest.raises(ValueError):
            trainer.train_model(wrong_data, config={'epochs': 5})


class TestEnhancedModelTrainerFallbackLogic:
    """Test fallback logic and production utilities."""
    
    @pytest.mark.unit
    @pytest.mark.model
    @patch('core.enhanced_trainer.get_logger')
    def test_logger_fallback(self, mock_get_logger):
        """Test logger initialization fallback."""
        mock_get_logger.side_effect = ImportError("Logger not available")
        
        with patch('logging.getLogger') as mock_logging:
            mock_logging.return_value = Mock()
            trainer = EnhancedModelTrainer()
            assert trainer is not None
    
    @pytest.mark.unit
    @pytest.mark.model
    @patch('core.enhanced_trainer.ModelDefaults')
    def test_constants_fallback(self, mock_defaults):
        """Test handling when model defaults are not available."""
        mock_defaults.HIDDEN_DIMS = [32, 16, 32]
        
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(input_dim=10)
        
        assert model.hidden_dims == [32, 16, 32]
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_graceful_degradation_on_save_error(self, normal_data_sample, test_data_dir):
        """Test graceful handling of save errors."""
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(normal_data_sample.shape[1], [8, 6, 8])
        model.train(normal_data_sample, epochs=3)
        
        # Try to save to invalid path
        invalid_path = "/invalid/path/model.pkl"
        
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            trainer.save_model(invalid_path)


class TestEnhancedModelTrainerIntegration:
    """Integration tests for EnhancedModelTrainer."""
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_complete_training_workflow(self, normal_data_sample, anomalous_data_sample):
        """Test complete training and evaluation workflow."""
        trainer = EnhancedModelTrainer()
        
        # 1. Create model
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [16, 8, 16])
        
        # 2. Split data
        split_idx = len(normal_data_sample) // 2
        train_data = normal_data_sample[:split_idx]
        val_data = normal_data_sample[split_idx:]
        
        # 3. Train model
        config = {
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 16,
            'early_stopping_patience': 5
        }
        
        training_history = trainer.train_model(train_data, val_data, config)
        
        # 4. Evaluate on normal data
        normal_metrics = trainer.evaluate_model(val_data)
        
        # 5. Evaluate on anomalous data
        anomalous_metrics = trainer.evaluate_model(anomalous_data_sample)
        
        # Verify workflow
        assert training_history is not None
        assert normal_metrics is not None
        assert anomalous_metrics is not None
        assert model.is_trained is True
        
        # Anomalous data should have higher reconstruction error
        assert (anomalous_metrics['mean_reconstruction_error'] > 
                normal_metrics['mean_reconstruction_error'])


@pytest.mark.slow
class TestEnhancedModelTrainerPerformance:
    """Performance tests for EnhancedModelTrainer."""
    
    @pytest.mark.benchmark
    def test_training_performance(self, normal_data_sample, benchmark):
        """Benchmark training performance."""
        trainer = EnhancedModelTrainer()
        input_dim = normal_data_sample.shape[1]
        
        def create_and_train():
            model = trainer.create_model(input_dim, [8, 6, 8])
            config = {'epochs': 5, 'learning_rate': 0.01, 'batch_size': 32}
            return trainer.train_model(normal_data_sample, config=config)
        
        result = benchmark(create_and_train)
        assert result is not None
    
    @pytest.mark.memory
    def test_memory_efficiency_large_model(self):
        """Test memory efficiency with large model."""
        trainer = EnhancedModelTrainer()
        
        # Large model
        input_dim = 100
        hidden_dims = [80, 60, 40, 60, 80]
        
        model = trainer.create_model(input_dim, hidden_dims)
        
        # Create large dataset
        large_data = np.random.randn(1000, input_dim)
        
        # Quick training
        config = {'epochs': 5, 'learning_rate': 0.01, 'batch_size': 64}
        training_history = trainer.train_model(large_data, config=config)
        
        assert training_history is not None
        assert model.is_trained is True
