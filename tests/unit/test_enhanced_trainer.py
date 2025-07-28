"""
Unit tests for EnhancedModelTrainer module.

Tests the model training, ProductionAutoencoder, and training management functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
from utils.constants import ModelDefaults


class TestProductionAutoencoder:
    """Test suite for ProductionAutoencoder class."""
    
    def test_initialization_default_hidden_dims(self):
        """Test autoencoder initialization with default hidden dimensions."""
        input_dim = 10
        autoencoder = ProductionAutoencoder(input_dim)
        
        assert autoencoder.input_dim == input_dim
        assert autoencoder.hidden_dims == ModelDefaults.HIDDEN_DIMS
        assert len(autoencoder.weights) > 0
        assert len(autoencoder.biases) > 0
        assert autoencoder.is_trained is False
        assert hasattr(autoencoder, 'logger')
    
    def test_initialization_custom_hidden_dims(self):
        """Test autoencoder initialization with custom hidden dimensions."""
        input_dim = 15
        hidden_dims = [8, 4, 2, 4, 8]
        
        autoencoder = ProductionAutoencoder(input_dim, hidden_dims)
        
        assert autoencoder.input_dim == input_dim
        assert autoencoder.hidden_dims == hidden_dims
        assert autoencoder.is_trained is False
    
    def test_weight_initialization_shapes(self):
        """Test that weight matrices have correct shapes."""
        input_dim = 10
        hidden_dims = [8, 4, 8]
        
        autoencoder = ProductionAutoencoder(input_dim, hidden_dims)
        
        # Expected layer dimensions: 10 -> 8 -> 4 -> 8 -> 10
        expected_shapes = [(10, 8), (8, 4), (4, 8), (8, 10)]
        
        assert len(autoencoder.weights) == len(expected_shapes)
        for weight, expected_shape in zip(autoencoder.weights, expected_shapes):
            assert weight.shape == expected_shape
    
    def test_bias_initialization_shapes(self):
        """Test that bias vectors have correct shapes."""
        input_dim = 10
        hidden_dims = [8, 4, 8]
        
        autoencoder = ProductionAutoencoder(input_dim, hidden_dims)
        
        # Expected bias dimensions: [8, 4, 8, 10]
        expected_shapes = [8, 4, 8, 10]
        
        assert len(autoencoder.biases) == len(expected_shapes)
        for bias, expected_shape in zip(autoencoder.biases, expected_shapes):
            assert bias.shape == (expected_shape,)
    
    @pytest.mark.unit
    def test_forward_pass_shape(self, normal_data_sample):
        """Test forward pass produces correct output shape."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        output = autoencoder._forward(normal_data_sample)
        
        assert output.shape == normal_data_sample.shape
        assert isinstance(output, np.ndarray)
    
    @pytest.mark.unit
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        input_dim = 5
        autoencoder = ProductionAutoencoder(input_dim, [3, 2, 3])
        
        single_sample = np.random.randn(1, input_dim)
        output = autoencoder._forward(single_sample)
        
        assert output.shape == (1, input_dim)
    
    @pytest.mark.unit
    def test_train_method_updates_weights(self, normal_data_sample):
        """Test that training updates model weights."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim, [5, 3, 5])
        
        # Store initial weights
        initial_weights = [w.copy() for w in autoencoder.weights]
        
        # Train for a few epochs
        autoencoder.train(normal_data_sample, epochs=5, learning_rate=0.01)
        
        # Weights should be different after training
        for initial_w, current_w in zip(initial_weights, autoencoder.weights):
            assert not np.array_equal(initial_w, current_w)
        
        assert autoencoder.is_trained is True
    
    @pytest.mark.unit
    def test_train_method_returns_history(self, normal_data_sample):
        """Test that training returns history dictionary."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        history = autoencoder.train(normal_data_sample, epochs=3)
        
        assert isinstance(history, dict)
        assert 'losses' in history
        assert 'final_loss' in history
        assert 'epochs' in history
        assert 'total_time' in history
        
        assert len(history['losses']) == 3  # 3 epochs
        assert history['epochs'] == 3
        assert history['total_time'] > 0
    
    @pytest.mark.unit
    def test_train_method_with_validation(self, normal_data_sample):
        """Test training with validation data."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        # Split data for training and validation
        train_data = normal_data_sample[:80]
        val_data = normal_data_sample[80:]
        
        history = autoencoder.train(train_data, val_data=val_data, epochs=3)
        
        assert 'val_losses' in history
        assert 'final_val_loss' in history
        assert len(history['val_losses']) == 3
    
    @pytest.mark.unit
    def test_train_method_early_stopping(self, normal_data_sample):
        """Test early stopping functionality."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        # Use validation data and early stopping
        train_data = normal_data_sample[:80]
        val_data = normal_data_sample[80:]
        
        history = autoencoder.train(
            train_data, 
            val_data=val_data, 
            epochs=20, 
            early_stopping_patience=3
        )
        
        # Should stop early if validation loss doesn't improve
        assert history['epochs'] <= 20
        assert 'early_stopped' in history
    
    @pytest.mark.unit
    def test_predict_method(self, normal_data_sample):
        """Test prediction method."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        predictions = autoencoder.predict(normal_data_sample)
        
        assert predictions.shape == normal_data_sample.shape
        assert isinstance(predictions, np.ndarray)
    
    @pytest.mark.unit
    def test_reconstruction_error(self, normal_data_sample):
        """Test reconstruction error calculation."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        errors = autoencoder.reconstruction_error(normal_data_sample)
        
        assert len(errors) == len(normal_data_sample)
        assert all(error >= 0 for error in errors)  # Errors should be non-negative
        assert isinstance(errors, np.ndarray)
    
    @pytest.mark.unit
    def test_save_and_load_model(self, normal_data_sample, temp_model_file):
        """Test model saving and loading."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        # Train the model
        autoencoder.train(normal_data_sample, epochs=2)
        
        # Save model
        autoencoder.save_model(str(temp_model_file))
        assert temp_model_file.exists()
        
        # Load model
        loaded_autoencoder = ProductionAutoencoder(input_dim)
        loaded_autoencoder.load_model(str(temp_model_file))
        
        # Check that loaded model produces same results
        original_output = autoencoder.predict(normal_data_sample[:5])
        loaded_output = loaded_autoencoder.predict(normal_data_sample[:5])
        
        np.testing.assert_array_almost_equal(original_output, loaded_output)
    
    @pytest.mark.unit
    def test_model_reproducibility(self, normal_data_sample):
        """Test that models are reproducible with same random seed."""
        input_dim = normal_data_sample.shape[1]
        
        # Train two models with same random seed
        np.random.seed(42)
        autoencoder1 = ProductionAutoencoder(input_dim)
        history1 = autoencoder1.train(normal_data_sample, epochs=3)
        
        np.random.seed(42)
        autoencoder2 = ProductionAutoencoder(input_dim)
        history2 = autoencoder2.train(normal_data_sample, epochs=3)
        
        # Should produce same results
        np.testing.assert_array_almost_equal(
            autoencoder1.predict(normal_data_sample[:5]),
            autoencoder2.predict(normal_data_sample[:5])
        )
    
    @pytest.mark.unit
    def test_empty_data_handling(self):
        """Test handling of empty input data."""
        autoencoder = ProductionAutoencoder(5)
        empty_data = np.array([]).reshape(0, 5)
        
        # Should handle empty data gracefully
        with pytest.raises(Exception):  # Should raise some error
            autoencoder.train(empty_data)
    
    @pytest.mark.unit
    def test_single_sample_training(self):
        """Test training with single sample."""
        input_dim = 5
        autoencoder = ProductionAutoencoder(input_dim)
        single_sample = np.random.randn(1, input_dim)
        
        history = autoencoder.train(single_sample, epochs=2)
        
        assert history['epochs'] == 2
        assert autoencoder.is_trained is True
    
    @pytest.mark.unit
    def test_different_learning_rates(self, normal_data_sample, training_configs):
        """Test training with different learning rates."""
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim)
        
        learning_rate = training_configs['learning_rate']
        epochs = training_configs['epochs']
        
        history = autoencoder.train(
            normal_data_sample, 
            epochs=epochs, 
            learning_rate=learning_rate
        )
        
        assert history['epochs'] == epochs
        assert autoencoder.is_trained is True


class TestEnhancedModelTrainer:
    """Test suite for EnhancedModelTrainer class."""
    
    def test_initialization(self):
        """Test EnhancedModelTrainer initialization."""
        trainer = EnhancedModelTrainer()
        
        assert trainer.model is None
        assert trainer.training_history == []
        assert hasattr(trainer, 'logger')
    
    @pytest.mark.unit
    def test_create_model(self):
        """Test model creation."""
        trainer = EnhancedModelTrainer()
        
        input_dim = 10
        hidden_dims = [8, 4, 8]
        
        model = trainer.create_model(input_dim, hidden_dims)
        
        assert isinstance(model, ProductionAutoencoder)
        assert model.input_dim == input_dim
        assert model.hidden_dims == hidden_dims
        assert trainer.model == model
    
    @pytest.mark.unit
    def test_create_model_default_hidden_dims(self):
        """Test model creation with default hidden dimensions."""
        trainer = EnhancedModelTrainer()
        
        model = trainer.create_model(input_dim=10)
        
        assert isinstance(model, ProductionAutoencoder)
        assert model.hidden_dims == ModelDefaults.HIDDEN_DIMS
    
    @pytest.mark.unit
    def test_train_model_success(self, normal_data_sample, mock_training_config):
        """Test successful model training."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        
        history = trainer.train_model(
            train_data=normal_data_sample,
            config=mock_training_config
        )
        
        assert isinstance(history, dict)
        assert 'final_train_loss' in history
        assert 'total_time' in history
        assert 'epochs' in history
        assert model.is_trained is True
    
    @pytest.mark.unit
    def test_train_model_with_validation(self, normal_data_sample, mock_training_config):
        """Test model training with validation data."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        
        # Split data
        train_data = normal_data_sample[:80]
        val_data = normal_data_sample[80:]
        
        history = trainer.train_model(
            train_data=train_data,
            val_data=val_data,
            config=mock_training_config
        )
        
        assert 'final_val_loss' in history
        assert 'val_losses' in history
    
    @pytest.mark.unit
    def test_train_model_no_model_created(self, normal_data_sample, mock_training_config):
        """Test training fails when no model is created."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model created"):
            trainer.train_model(normal_data_sample, config=mock_training_config)
    
    @pytest.mark.unit
    def test_evaluate_model(self, normal_data_sample):
        """Test model evaluation."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        
        # Train briefly
        model.train(normal_data_sample, epochs=2)
        
        metrics = trainer.evaluate_model(normal_data_sample)
        
        assert isinstance(metrics, dict)
        assert 'mean_reconstruction_error' in metrics
        assert 'std_reconstruction_error' in metrics
        assert 'min_reconstruction_error' in metrics
        assert 'max_reconstruction_error' in metrics
    
    @pytest.mark.unit
    def test_evaluate_model_no_model(self, normal_data_sample):
        """Test evaluation fails when no model exists."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model created"):
            trainer.evaluate_model(normal_data_sample)
    
    @pytest.mark.unit
    def test_save_model(self, normal_data_sample, temp_model_file):
        """Test model saving."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        model.train(normal_data_sample, epochs=2)
        
        trainer.save_model(str(temp_model_file))
        
        assert temp_model_file.exists()
    
    @pytest.mark.unit
    def test_save_model_no_model(self, temp_model_file):
        """Test saving fails when no model exists."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model created"):
            trainer.save_model(str(temp_model_file))
    
    @pytest.mark.unit
    def test_load_model(self, normal_data_sample, temp_model_file):
        """Test model loading."""
        # First create and save a model
        trainer1 = EnhancedModelTrainer()
        input_dim = normal_data_sample.shape[1]
        model1 = trainer1.create_model(input_dim)
        model1.train(normal_data_sample, epochs=2)
        trainer1.save_model(str(temp_model_file))
        
        # Now load in new trainer
        trainer2 = EnhancedModelTrainer()
        trainer2.load_model(str(temp_model_file), input_dim)
        
        assert trainer2.model is not None
        assert isinstance(trainer2.model, ProductionAutoencoder)
        
        # Should produce similar results
        output1 = model1.predict(normal_data_sample[:5])
        output2 = trainer2.model.predict(normal_data_sample[:5])
        np.testing.assert_array_almost_equal(output1, output2)
    
    @pytest.mark.unit
    def test_get_model_info(self, normal_data_sample):
        """Test getting model information."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        hidden_dims = [8, 4, 8]
        model = trainer.create_model(input_dim, hidden_dims)
        
        info = trainer.get_model_info()
        
        assert isinstance(info, dict)
        assert 'input_dim' in info
        assert 'hidden_dims' in info
        assert 'is_trained' in info
        assert 'total_parameters' in info
        
        assert info['input_dim'] == input_dim
        assert info['hidden_dims'] == hidden_dims
        assert info['is_trained'] is False
    
    @pytest.mark.unit
    def test_get_model_info_no_model(self):
        """Test getting model info when no model exists."""
        trainer = EnhancedModelTrainer()
        
        with pytest.raises(ValueError, match="No model created"):
            trainer.get_model_info()
    
    @pytest.mark.unit
    def test_training_history_tracking(self, normal_data_sample, mock_training_config):
        """Test that training history is properly tracked."""
        trainer = EnhancedModelTrainer()
        
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        
        # Train multiple times
        history1 = trainer.train_model(normal_data_sample, config=mock_training_config)
        history2 = trainer.train_model(normal_data_sample, config=mock_training_config)
        
        assert len(trainer.training_history) == 2
        assert trainer.training_history[0] == history1
        assert trainer.training_history[1] == history2
    
    @pytest.mark.unit
    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        trainer = EnhancedModelTrainer()
        
        input_dim = 10
        hidden_dims = [8, 4, 8]
        model = trainer.create_model(input_dim, hidden_dims)
        
        info = trainer.get_model_info()
        
        # Calculate expected parameters
        # Weights: 10*8 + 8*4 + 4*8 + 8*10 = 80 + 32 + 32 + 80 = 224
        # Biases: 8 + 4 + 8 + 10 = 30
        # Total: 224 + 30 = 254
        expected_params = 254
        
        assert info['total_parameters'] == expected_params


class TestEnhancedModelTrainerIntegration:
    """Integration tests for EnhancedModelTrainer with complete workflows."""
    
    @pytest.mark.integration
    def test_complete_training_workflow(self, normal_data_sample, mock_training_config):
        """Test complete training workflow from creation to evaluation."""
        trainer = EnhancedModelTrainer()
        
        # Step 1: Create model
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim)
        assert model is not None
        
        # Step 2: Train model
        history = trainer.train_model(normal_data_sample, config=mock_training_config)
        assert model.is_trained
        
        # Step 3: Evaluate model
        metrics = trainer.evaluate_model(normal_data_sample)
        assert metrics['mean_reconstruction_error'] >= 0
        
        # Step 4: Get model info
        info = trainer.get_model_info()
        assert info['is_trained'] is True
    
    @pytest.mark.integration
    def test_model_persistence_workflow(self, normal_data_sample, mock_training_config, temp_model_file):
        """Test complete model persistence workflow."""
        # Train and save model
        trainer1 = EnhancedModelTrainer()
        input_dim = normal_data_sample.shape[1]
        
        model1 = trainer1.create_model(input_dim)
        trainer1.train_model(normal_data_sample, config=mock_training_config)
        trainer1.save_model(str(temp_model_file))
        
        # Load model in new trainer
        trainer2 = EnhancedModelTrainer()
        trainer2.load_model(str(temp_model_file), input_dim)
        
        # Both should produce same results
        test_data = normal_data_sample[:10]
        output1 = trainer1.model.predict(test_data)
        output2 = trainer2.model.predict(test_data)
        
        np.testing.assert_array_almost_equal(output1, output2, decimal=6)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_with_large_dataset(self, large_dataset):
        """Test training performance with large datasets."""
        import time
        
        trainer = EnhancedModelTrainer()
        
        # Convert to numpy array and select features
        feature_cols = [col for col in large_dataset.columns if col != 'class']
        data_array = large_dataset[feature_cols].values
        
        start_time = time.time()
        
        # Create and train model
        model = trainer.create_model(data_array.shape[1])
        
        config = {
            'epochs': 5,  # Reduced for performance test
            'learning_rate': 0.001,
            'batch_size': 64
        }
        
        history = trainer.train_model(data_array, config=config)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 60.0  # 1 minute max
        assert model.is_trained
        assert history['epochs'] == 5
