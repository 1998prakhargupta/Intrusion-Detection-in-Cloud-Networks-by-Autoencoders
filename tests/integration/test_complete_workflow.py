"""
Integration tests for the complete NIDS autoencoder workflow.

Tests the integration between DataLoader, DataPreprocessor, EnhancedModelTrainer, 
ProductionAutoencoder, and ModelEvaluator components.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from core.enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
from core.evaluator import ModelEvaluator


class TestCompleteWorkflow:
    """Test complete end-to-end workflow integration."""
    
    @pytest.mark.integration
    def test_basic_training_workflow(self, sample_dataset, test_data_dir):
        """Test basic training workflow from data loading to evaluation."""
        # Create test data file
        data_file = test_data_dir / "test_network_data.csv"
        sample_dataset.to_csv(data_file, index=False)
        
        # Step 1: Data Loading
        loader = DataLoader()
        data, features, labels = loader.load_and_validate_data(str(data_file))
        
        assert data is not None
        assert features is not None
        assert labels is not None
        assert len(data) > 0
        
        # Step 2: Data Preprocessing
        preprocessor = DataPreprocessor()
        processed_features, feature_names = preprocessor.preprocess_features(features)
        
        assert processed_features.shape[0] == features.shape[0]
        assert len(feature_names) > 0
        
        # Step 3: Data Preparation for Training
        normal_data, _ = preprocessor.separate_normal_anomalous(processed_features, labels)
        train_data, validation_data = preprocessor.prepare_training_data(normal_data)
        
        assert len(train_data) > 0
        assert len(validation_data) > 0
        
        # Step 4: Model Training
        input_dim = train_data.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        
        trainer = EnhancedModelTrainer(autoencoder)
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=5,  # Quick training for test
            learning_rate=0.001,
            batch_size=32
        )
        
        assert autoencoder.is_trained
        assert 'train_loss' in training_history
        assert len(training_history['train_loss']) > 0
        
        # Step 5: Model Evaluation
        evaluator = ModelEvaluator()
        
        # Create some anomalous data for evaluation
        anomalous_features = processed_features[labels == 'anomaly'] if 'anomaly' in labels.values else None
        
        evaluation_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=normal_data[:100],  # Use subset for faster testing
            anomalous_data=anomalous_features[:20] if anomalous_features is not None else None
        )
        
        assert 'normal_errors' in evaluation_results
        assert 'thresholds' in evaluation_results
        
        # Step 6: Generate Report
        if anomalous_features is not None:
            report = evaluator.generate_evaluation_report()
            assert 'summary' in report
            assert 'model_performance' in report
    
    @pytest.mark.integration
    def test_model_persistence_workflow(self, normal_data_sample, test_data_dir):
        """Test model training, saving, loading, and evaluation workflow."""
        # Train model
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        
        trainer = EnhancedModelTrainer(autoencoder)
        
        # Quick training
        train_data, validation_data = np.split(normal_data_sample, [int(0.8 * len(normal_data_sample))])
        
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=3,
            learning_rate=0.01,
            batch_size=16
        )
        
        assert autoencoder.is_trained
        
        # Save model
        model_path = test_data_dir / "trained_autoencoder.pkl"
        trainer.save_model(str(model_path))
        assert model_path.exists()
        
        # Load model in new trainer
        new_trainer = EnhancedModelTrainer()
        loaded_model = new_trainer.load_model(str(model_path))
        
        assert loaded_model.is_trained
        assert loaded_model.input_dim == input_dim
        
        # Evaluate both models and compare
        evaluator = ModelEvaluator()
        
        original_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        loaded_results = evaluator.evaluate_model(
            model=loaded_model,
            normal_data=validation_data
        )
        
        # Results should be very similar (allowing for small numerical differences)
        original_mean = original_results['normal_errors']['mean']
        loaded_mean = loaded_results['normal_errors']['mean']
        assert abs(original_mean - loaded_mean) < 0.001
    
    @pytest.mark.integration
    def test_data_preprocessing_pipeline(self, sample_dataset, test_data_dir):
        """Test complete data preprocessing pipeline."""
        # Create test data file
        data_file = test_data_dir / "test_preprocessing_data.csv"
        sample_dataset.to_csv(data_file, index=False)
        
        # Load data
        loader = DataLoader()
        data, features, labels = loader.load_and_validate_data(str(data_file))
        
        # Get data summary
        summary = loader.get_data_summary(data, labels)
        assert 'total_samples' in summary
        assert 'feature_count' in summary
        
        # Preprocess features
        preprocessor = DataPreprocessor()
        processed_features, feature_names = preprocessor.preprocess_features(features)
        
        # Check preprocessing results
        assert not np.isnan(processed_features).any(), "Processed features should not contain NaN"
        assert not np.isinf(processed_features).any(), "Processed features should not contain Inf"
        
        # Scale data
        scaled_features = preprocessor.scale_data(processed_features)
        
        # Check scaling results
        feature_means = np.mean(scaled_features, axis=0)
        feature_stds = np.std(scaled_features, axis=0)
        
        # Should be approximately standard normal
        assert np.allclose(feature_means, 0, atol=1e-10), "Scaled features should have zero mean"
        assert np.allclose(feature_stds, 1, atol=1e-10), "Scaled features should have unit variance"
        
        # Separate normal and anomalous data
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(scaled_features, labels)
        
        assert len(normal_data) > 0, "Should have normal data"
        total_expected = len(normal_data) + (len(anomalous_data) if anomalous_data is not None else 0)
        assert total_expected == len(scaled_features), "All data should be accounted for"
    
    @pytest.mark.integration 
    def test_training_with_different_configurations(self, normal_data_sample):
        """Test training with different hyperparameter configurations."""
        input_dim = normal_data_sample.shape[1]
        
        # Split data
        train_data, validation_data = np.split(normal_data_sample, [int(0.8 * len(normal_data_sample))])
        
        configurations = [
            {'learning_rate': 0.01, 'batch_size': 16, 'hidden_dims': [8, 4]},
            {'learning_rate': 0.001, 'batch_size': 32, 'hidden_dims': [6, 3]},
            {'learning_rate': 0.005, 'batch_size': 8, 'hidden_dims': [10, 5]}
        ]
        
        results = []
        
        for config in configurations:
            # Create new autoencoder with specific architecture
            autoencoder = ProductionAutoencoder(
                input_dim=input_dim,
                hidden_dims=config['hidden_dims']
            )
            
            trainer = EnhancedModelTrainer(autoencoder)
            
            # Train with specific configuration
            training_history = trainer.train(
                train_data=train_data,
                validation_data=validation_data,
                epochs=3,  # Quick training for test
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size']
            )
            
            assert autoencoder.is_trained
            
            # Evaluate
            evaluator = ModelEvaluator()
            eval_results = evaluator.evaluate_model(
                model=autoencoder,
                normal_data=validation_data
            )
            
            results.append({
                'config': config,
                'final_loss': training_history['train_loss'][-1],
                'mean_error': eval_results['normal_errors']['mean']
            })
        
        # All configurations should produce valid results
        assert len(results) == len(configurations)
        for result in results:
            assert result['final_loss'] > 0
            assert result['mean_error'] > 0
    
    @pytest.mark.integration
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid data file
        loader = DataLoader()
        
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            loader.load_and_validate_data("nonexistent_file.csv")
        
        # Test with untrained model
        autoencoder = ProductionAutoencoder(input_dim=10)
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            evaluator.evaluate_model(
                model=autoencoder,
                normal_data=np.random.randn(100, 10)
            )
        
        # Test with mismatched dimensions
        autoencoder = ProductionAutoencoder(input_dim=5)
        trainer = EnhancedModelTrainer(autoencoder)
        
        with pytest.raises(ValueError):
            trainer.train(
                train_data=np.random.randn(100, 10),  # Wrong dimension
                validation_data=np.random.randn(20, 10),
                epochs=1
            )
    
    @pytest.mark.integration
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with larger datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger dataset
        large_normal_data = np.random.randn(2000, 20)
        
        # Train model
        autoencoder = ProductionAutoencoder(input_dim=20, hidden_dims=[10, 5])
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(large_normal_data, [int(0.8 * len(large_normal_data))])
        
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=3,
            batch_size=64  # Larger batch size for efficiency
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.2f}MB"
        
        # Results should still be valid
        assert autoencoder.is_trained
        assert 'normal_errors' in evaluation_results
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_benchmarking(self, normal_data_sample):
        """Test performance benchmarking of the complete pipeline."""
        import time
        
        # Measure training time
        start_time = time.time()
        
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(normal_data_sample, [int(0.8 * len(normal_data_sample))])
        
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=10,
            learning_rate=0.01,
            batch_size=32
        )
        
        training_time = time.time() - start_time
        
        # Measure evaluation time
        start_time = time.time()
        
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        evaluation_time = time.time() - start_time
        
        # Performance assertions
        assert training_time < 30.0, f"Training took too long: {training_time:.2f}s"
        assert evaluation_time < 5.0, f"Evaluation took too long: {evaluation_time:.2f}s"
        
        # Quality assertions
        assert autoencoder.is_trained
        assert evaluation_results['normal_errors']['mean'] > 0
        
        print(f"Performance metrics:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Evaluation time: {evaluation_time:.2f}s")
        print(f"  Final training loss: {training_history['train_loss'][-1]:.6f}")
        print(f"  Mean reconstruction error: {evaluation_results['normal_errors']['mean']:.6f}")


class TestComponentIntegration:
    """Test integration between specific components."""
    
    @pytest.mark.integration
    def test_loader_preprocessor_integration(self, sample_dataset, test_data_dir):
        """Test integration between DataLoader and DataPreprocessor."""
        # Create test data file
        data_file = test_data_dir / "integration_test_data.csv"
        sample_dataset.to_csv(data_file, index=False)
        
        # Load data
        loader = DataLoader()
        data, features, labels = loader.load_and_validate_data(str(data_file))
        
        # Process with preprocessor
        preprocessor = DataPreprocessor()
        processed_features, feature_names = preprocessor.preprocess_features(features)
        
        # Verify integration
        assert processed_features.shape[0] == features.shape[0]
        assert len(feature_names) <= features.shape[1]  # May be reduced due to preprocessing
        
        # Test that processed features can be used for scaling
        scaled_features = preprocessor.scale_data(processed_features)
        assert scaled_features.shape == processed_features.shape
    
    @pytest.mark.integration
    def test_trainer_evaluator_integration(self, normal_data_sample):
        """Test integration between EnhancedModelTrainer and ModelEvaluator."""
        # Train model
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(normal_data_sample, [int(0.8 * len(normal_data_sample))])
        
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=5
        )
        
        # Evaluate model
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        # Verify integration
        assert autoencoder.is_trained
        assert 'normal_errors' in evaluation_results
        
        # Training history should correlate with evaluation results
        final_train_loss = training_history['train_loss'][-1]
        mean_eval_error = evaluation_results['normal_errors']['mean']
        
        # Both should be in reasonable ranges
        assert 0 < final_train_loss < 1.0
        assert 0 < mean_eval_error < 1.0
    
    @pytest.mark.integration
    def test_autoencoder_evaluator_consistency(self, normal_data_sample):
        """Test consistency between autoencoder reconstruction and evaluator metrics."""
        # Train model
        input_dim = normal_data_sample.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, test_data = np.split(normal_data_sample, [int(0.8 * len(normal_data_sample))])
        
        trainer.train(
            train_data=train_data,
            validation_data=test_data,
            epochs=5
        )
        
        # Calculate reconstruction errors directly
        direct_errors = autoencoder.reconstruction_error(test_data)
        
        # Calculate through evaluator
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=test_data
        )
        
        evaluator_errors = evaluation_results['normal_errors']['values']
        
        # Results should be identical
        np.testing.assert_array_almost_equal(direct_errors, evaluator_errors, decimal=10)
    
    @pytest.mark.integration
    def test_preprocessor_trainer_compatibility(self, sample_dataset, test_data_dir):
        """Test compatibility between preprocessor output and trainer input."""
        # Create and load data
        data_file = test_data_dir / "compatibility_test_data.csv"
        sample_dataset.to_csv(data_file, index=False)
        
        loader = DataLoader()
        data, features, labels = loader.load_and_validate_data(str(data_file))
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_features, feature_names = preprocessor.preprocess_features(features)
        scaled_features = preprocessor.scale_data(processed_features)
        
        normal_data, _ = preprocessor.separate_normal_anomalous(scaled_features, labels)
        train_data, validation_data = preprocessor.prepare_training_data(normal_data)
        
        # Train model with preprocessed data
        input_dim = train_data.shape[1]
        autoencoder = ProductionAutoencoder(input_dim=input_dim)
        trainer = EnhancedModelTrainer(autoencoder)
        
        # This should work without errors
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=3
        )
        
        assert autoencoder.is_trained
        assert len(training_history['train_loss']) > 0


class TestRobustnessAndEdgeCases:
    """Test system robustness and edge cases."""
    
    @pytest.mark.integration
    def test_minimal_dataset_handling(self):
        """Test handling of minimal datasets."""
        # Create minimal dataset
        minimal_data = np.random.randn(10, 5)  # Very small dataset
        
        # Should still be able to train (though not well)
        autoencoder = ProductionAutoencoder(input_dim=5)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = minimal_data[:8], minimal_data[8:]
        
        # Should handle small datasets gracefully
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=2,
            batch_size=4  # Small batch size
        )
        
        assert autoencoder.is_trained
        assert len(training_history['train_loss']) > 0
    
    @pytest.mark.integration
    def test_single_sample_evaluation(self):
        """Test evaluation with single samples."""
        # Train a model first
        normal_data = np.random.randn(100, 5)
        autoencoder = ProductionAutoencoder(input_dim=5)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(normal_data, [80])
        trainer.train(train_data=train_data, validation_data=validation_data, epochs=3)
        
        # Evaluate with single sample
        single_sample = normal_data[0:1]  # Keep as 2D array
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=single_sample
        )
        
        assert 'normal_errors' in results
        assert len(results['normal_errors']['values']) == 1
    
    @pytest.mark.integration
    def test_identical_data_handling(self):
        """Test handling of identical data points."""
        # Create dataset with identical values
        identical_data = np.ones((50, 5)) * 0.5  # All samples identical
        
        autoencoder = ProductionAutoencoder(input_dim=5)
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(identical_data, [40])
        
        # Should handle without crashing
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=3
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        # Should produce consistent results
        errors = results['normal_errors']['values']
        assert np.std(errors) < 0.1  # Errors should be very similar
    
    @pytest.mark.integration
    def test_high_dimensional_data(self):
        """Test handling of high-dimensional data."""
        # Create high-dimensional dataset
        high_dim_data = np.random.randn(200, 50)  # 50 features
        
        autoencoder = ProductionAutoencoder(
            input_dim=50,
            hidden_dims=[25, 10, 5]  # Deeper architecture
        )
        trainer = EnhancedModelTrainer(autoencoder)
        
        train_data, validation_data = np.split(high_dim_data, [160])
        
        training_history = trainer.train(
            train_data=train_data,
            validation_data=validation_data,
            epochs=5,
            batch_size=16
        )
        
        assert autoencoder.is_trained
        
        # Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model=autoencoder,
            normal_data=validation_data
        )
        
        assert 'normal_errors' in results
        assert len(results['normal_errors']['values']) == len(validation_data)
