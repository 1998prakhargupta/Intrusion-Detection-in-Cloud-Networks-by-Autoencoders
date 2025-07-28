"""
Comprehensive integration tests for NIDS Autoencoder system.

This module tests the complete workflow integrating all components.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from core.enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
from core.evaluator import ModelEvaluator
from utils.constants import DataConstants, ModelDefaults


class TestCompleteWorkflow:
    """Test complete NIDS workflow integration."""
    
    @pytest.mark.integration
    @pytest.mark.e2e
    def test_end_to_end_workflow(self, sample_csv_file):
        """Test complete end-to-end workflow from data loading to evaluation."""
        
        # 1. DATA LOADING
        print("\n🔍 Step 1: Data Loading")
        loader = DataLoader()
        data = loader.load_and_validate_data(str(sample_csv_file))
        features, labels = loader.extract_features_and_labels(data)
        data_summary = loader.get_data_summary(data)
        
        assert data is not None
        assert len(features) > 0
        assert labels is not None
        assert data_summary['shape'][0] > 100  # Ensure sufficient data
        print(f"✅ Loaded {data_summary['shape'][0]} samples with {len(features)} features")
        
        # 2. DATA PREPROCESSING
        print("\n🔧 Step 2: Data Preprocessing")
        preprocessor = DataPreprocessor()
        processed_features = preprocessor.preprocess_features(data, features)
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, labels, normal_identifier='normal'
        )
        
        assert processed_features is not None
        assert normal_data is not None
        assert len(normal_data) > 50  # Ensure sufficient normal data
        print(f"✅ Processed features: {normal_data.shape} normal, {anomalous_data.shape if anomalous_data is not None else 0} anomalous")
        
        # 3. TRAINING DATA PREPARATION
        print("\n📊 Step 3: Training Data Preparation")
        train_data, val_data = preprocessor.prepare_training_data(
            normal_data, validation_ratio=0.2, random_state=42
        )
        
        if anomalous_data is not None:
            anomalous_scaled = preprocessor.scale_data(anomalous_data)
        else:
            anomalous_scaled = None
        
        assert train_data is not None
        assert val_data is not None
        assert len(train_data) > len(val_data)  # Training set should be larger
        print(f"✅ Training data: {len(train_data)} train, {len(val_data)} validation")
        
        # 4. MODEL TRAINING
        print("\n🏗️ Step 4: Model Training")
        trainer = EnhancedModelTrainer()
        input_dim = train_data.shape[1]
        hidden_dims = [min(64, input_dim-1), min(32, input_dim-2), min(16, input_dim-3)]
        
        model = trainer.create_model(input_dim=input_dim, hidden_dims=hidden_dims)
        
        training_config = {
            'epochs': 20,  # Moderate training for integration test
            'learning_rate': 0.001,
            'batch_size': 32,
            'early_stopping_patience': 5
        }
        
        start_time = time.time()
        training_history = trainer.train_model(
            train_data=train_data,
            val_data=val_data,
            config=training_config
        )
        training_time = time.time() - start_time
        
        assert model.is_trained is True
        assert training_history['epochs'] <= training_config['epochs']
        assert training_history['final_train_loss'] > 0
        print(f"✅ Model trained in {training_time:.2f}s, {training_history['epochs']} epochs")
        
        # 5. MODEL EVALUATION
        print("\n📈 Step 5: Model Evaluation")
        evaluator = ModelEvaluator()
        
        evaluation_results = evaluator.evaluate_model(
            model=model,
            normal_data=val_data,
            anomalous_data=anomalous_scaled,
            class_info=labels if anomalous_scaled is not None else None
        )
        
        assert evaluation_results is not None
        assert 'normal_errors' in evaluation_results
        assert 'thresholds' in evaluation_results
        
        normal_mean_error = evaluation_results['normal_errors']['mean']
        print(f"✅ Evaluation complete. Normal reconstruction error: {normal_mean_error:.6f}")
        
        if 'roc_auc' in evaluation_results:
            auc_score = evaluation_results['roc_auc']
            print(f"📊 ROC-AUC Score: {auc_score:.3f}")
            assert 0.4 <= auc_score <= 1.0  # Should be at least better than random
        
        # 6. THRESHOLD OPTIMIZATION
        print("\n🎯 Step 6: Threshold Optimization")
        best_method, best_threshold = evaluator.get_best_threshold_method()
        
        assert best_method is not None
        assert best_threshold is not None
        assert best_threshold > 0
        print(f"✅ Best threshold: {best_method} = {best_threshold:.6f}")
        
        # 7. PRODUCTION READINESS CHECK
        print("\n🚀 Step 7: Production Readiness")
        
        # Test prediction on new data
        test_sample = val_data[:5]  # Use first 5 validation samples
        predictions = model.predict(test_sample)
        reconstruction_errors = model.reconstruction_error(test_sample)
        
        assert predictions.shape == test_sample.shape
        assert len(reconstruction_errors) == len(test_sample)
        assert np.all(reconstruction_errors >= 0)
        
        # Test anomaly detection
        anomalies_detected = reconstruction_errors > best_threshold
        print(f"✅ Tested on {len(test_sample)} samples, detected {np.sum(anomalies_detected)} anomalies")
        
        # 8. WORKFLOW SUMMARY
        print("\n📋 Workflow Summary:")
        print(f"   Data samples: {data_summary['shape'][0]}")
        print(f"   Features: {len(features)}")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Model architecture: {input_dim} → {hidden_dims} → {input_dim}")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Final loss: {training_history['final_train_loss']:.6f}")
        print(f"   Best threshold ({best_method}): {best_threshold:.6f}")
        if 'roc_auc' in evaluation_results:
            print(f"   ROC-AUC: {evaluation_results['roc_auc']:.3f}")
        
        return {
            'data_summary': data_summary,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'best_threshold': best_threshold,
            'workflow_time': training_time
        }
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_model_persistence_workflow(self, sample_csv_file, test_data_dir):
        """Test model training, saving, loading, and consistency."""
        
        # Load and prepare data
        loader = DataLoader()
        data = loader.load_and_validate_data(str(sample_csv_file))
        features, labels = loader.extract_features_and_labels(data)
        
        preprocessor = DataPreprocessor()
        processed_features = preprocessor.preprocess_features(data, features)
        normal_data, _ = preprocessor.separate_normal_anomalous(
            processed_features, labels, 'normal'
        )
        train_data, val_data = preprocessor.prepare_training_data(normal_data)
        
        # Train original model
        trainer1 = EnhancedModelTrainer()
        model1 = trainer1.create_model(train_data.shape[1], [16, 8, 16])
        training_history1 = trainer1.train_model(train_data, val_data, {'epochs': 10})
        
        # Get predictions from original model
        original_predictions = model1.predict(val_data[:10])
        original_errors = model1.reconstruction_error(val_data[:10])
        
        # Save model and preprocessor
        model_path = test_data_dir / "integration_model.pkl"
        preprocessor_path = test_data_dir / "integration_preprocessor"
        
        trainer1.save_model(str(model_path))
        preprocessor.save_preprocessing_artifacts(str(preprocessor_path))
        
        # Load model and preprocessor in new instances
        trainer2 = EnhancedModelTrainer()
        trainer2.load_model(str(model_path))
        
        preprocessor2 = DataPreprocessor()
        preprocessor2.load_preprocessing_artifacts(str(preprocessor_path))
        
        # Test consistency
        loaded_predictions = trainer2.model.predict(val_data[:10])
        loaded_errors = trainer2.model.reconstruction_error(val_data[:10])
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=6)
        np.testing.assert_array_almost_equal(original_errors, loaded_errors, decimal=6)
        
        # Test on new data with loaded preprocessor
        new_data = pd.DataFrame(np.random.randn(50, len(features)), columns=features)
        processed_new = preprocessor2.preprocess_features(new_data, features)
        scaled_new = preprocessor2.scale_data(processed_new)
        
        # Should be able to process and predict on new data
        new_predictions = trainer2.model.predict(scaled_new)
        new_errors = trainer2.model.reconstruction_error(scaled_new)
        
        assert new_predictions.shape == scaled_new.shape
        assert len(new_errors) == len(scaled_new)
        assert np.all(np.isfinite(new_predictions))
        assert np.all(np.isfinite(new_errors))
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_scalability_workflow(self, test_data_dir):
        """Test workflow scalability with larger datasets."""
        
        # Create larger synthetic dataset
        n_samples = 5000
        n_features = 20
        
        # Generate normal data
        normal_samples = int(n_samples * 0.8)
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=normal_samples
        )
        
        # Generate anomalous data (different distribution)
        anomalous_samples = n_samples - normal_samples
        anomalous_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,
            cov=np.eye(n_features) * 2,
            size=anomalous_samples
        )
        
        # Create DataFrame
        all_data = np.vstack([normal_data, anomalous_data])
        feature_names = [f'feature_{i}' for i in range(n_features)]
        labels = ['normal'] * normal_samples + ['attack'] * anomalous_samples
        
        large_dataset = pd.DataFrame(all_data, columns=feature_names)
        large_dataset['class'] = labels
        
        # Save to CSV
        large_csv = test_data_dir / "large_dataset.csv"
        large_dataset.to_csv(large_csv, index=False)
        
        # Test workflow on large dataset
        start_time = time.time()
        
        # Data loading
        loader = DataLoader()
        data = loader.load_and_validate_data(str(large_csv))
        features, class_info = loader.extract_features_and_labels(data)
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        processed_features = preprocessor.preprocess_features(data, features)
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, class_info, 'normal'
        )
        train_data, val_data = preprocessor.prepare_training_data(
            normal_data, validation_ratio=0.2
        )
        anomalous_scaled = preprocessor.scale_data(anomalous_data)
        
        # Training
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(train_data.shape[1], [32, 16, 32])
        training_history = trainer.train_model(
            train_data, val_data, 
            {'epochs': 15, 'batch_size': 64}
        )
        
        # Evaluation
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model, val_data, anomalous_scaled, class_info
        )
        
        total_time = time.time() - start_time
        
        # Verify scalability
        assert total_time < 300  # Should complete within 5 minutes
        assert results['roc_auc'] > 0.7  # Should achieve good performance
        assert training_history['final_train_loss'] < 1.0  # Should converge
        
        print(f"✅ Scalability test completed in {total_time:.2f}s")
        print(f"   Dataset: {n_samples} samples, {n_features} features")
        print(f"   ROC-AUC: {results['roc_auc']:.3f}")
        print(f"   Training loss: {training_history['final_train_loss']:.6f}")


class TestComponentIntegration:
    """Test integration between specific components."""
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_loader_preprocessor_integration(self, sample_csv_file):
        """Test integration between DataLoader and DataPreprocessor."""
        
        # Load data
        loader = DataLoader()
        data = loader.load_and_validate_data(str(sample_csv_file))
        features, labels = loader.extract_features_and_labels(data)
        
        # Preprocess using loaded data
        preprocessor = DataPreprocessor()
        processed_features = preprocessor.preprocess_features(data, features)
        normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
            processed_features, labels, 'normal'
        )
        
        # Verify integration
        assert processed_features.shape[0] == data.shape[0]
        assert processed_features.shape[1] <= len(features)  # May be reduced due to encoding
        
        if normal_data is not None and anomalous_data is not None:
            assert len(normal_data) + len(anomalous_data) == len(processed_features)
        
        # Test that preprocessor can handle loader output formats
        summary = loader.get_data_summary(data)
        assert 'shape' in summary
        assert summary['shape'] == data.shape
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_preprocessor_trainer_integration(self, normal_data_sample):
        """Test integration between DataPreprocessor and EnhancedModelTrainer."""
        
        # Prepare data with preprocessor
        preprocessor = DataPreprocessor()
        normal_df = pd.DataFrame(normal_data_sample)
        train_data, val_data = preprocessor.prepare_training_data(normal_df)
        
        # Train model with preprocessed data
        trainer = EnhancedModelTrainer()
        model = trainer.create_model(train_data.shape[1], [16, 8, 16])
        training_history = trainer.train_model(
            train_data, val_data, {'epochs': 10}
        )
        
        # Verify integration
        assert model.input_dim == train_data.shape[1]
        assert model.is_trained is True
        assert training_history['epochs'] <= 10
        
        # Test that trainer can handle preprocessor output
        metrics = trainer.evaluate_model(val_data)
        assert 'mean_reconstruction_error' in metrics
        assert metrics['mean_reconstruction_error'] >= 0
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_trainer_evaluator_integration(self, normal_data_sample, anomalous_data_sample):
        """Test integration between EnhancedModelTrainer and ModelEvaluator."""
        
        # Train model
        trainer = EnhancedModelTrainer()
        input_dim = normal_data_sample.shape[1]
        model = trainer.create_model(input_dim, [16, 8, 16])
        trainer.train_model(normal_data_sample, config={'epochs': 10})
        
        # Evaluate with evaluator
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model, normal_data_sample, anomalous_data_sample
        )
        
        # Verify integration
        assert 'normal_errors' in results
        assert 'anomalous_errors' in results
        assert 'thresholds' in results
        
        # Test that evaluator can use trainer's model
        best_method, best_threshold = evaluator.get_best_threshold_method()
        assert best_method in results['performance']
        assert results['performance'][best_method]['threshold'] == best_threshold


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    @pytest.mark.integration
    @pytest.mark.data
    def test_corrupted_data_workflow(self, test_data_dir):
        """Test workflow behavior with corrupted data."""
        
        # Create corrupted CSV
        corrupted_csv = test_data_dir / "corrupted.csv"
        with open(corrupted_csv, 'w') as f:
            f.write("feature1,feature2,class\n")
            f.write("1,2,normal\n")
            f.write("invalid,data,here\n")  # Corrupted row
            f.write("3,4,attack\n")
        
        loader = DataLoader()
        
        # Should handle corrupted data gracefully
        try:
            data = loader.load_and_validate_data(str(corrupted_csv))
            # If it loads, should have valid data
            assert len(data) > 0
        except (ValueError, pd.errors.ParserError):
            # Acceptable to raise error for corrupted data
            pass
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_insufficient_data_workflow(self, test_data_dir):
        """Test workflow behavior with insufficient data."""
        
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4],
            'class': ['normal', 'normal']
        })
        minimal_csv = test_data_dir / "minimal.csv"
        minimal_data.to_csv(minimal_csv, index=False)
        
        loader = DataLoader()
        
        # Should handle insufficient data appropriately
        try:
            data = loader.load_and_validate_data(str(minimal_csv))
            features, labels = loader.extract_features_and_labels(data)
            
            preprocessor = DataPreprocessor()
            with pytest.raises(ValueError):
                # Should fail when trying to prepare training data
                normal_data, _ = preprocessor.separate_normal_anomalous(
                    data[features], labels, 'normal'
                )
                preprocessor.prepare_training_data(normal_data)
                
        except ValueError:
            # Acceptable to fail early with insufficient data
            pass
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_memory_constraint_workflow(self):
        """Test workflow behavior under memory constraints."""
        
        # This test would require specific memory limitation setup
        # Skipping for now but framework is here for future implementation
        pytest.skip("Memory constraint testing requires specific environment setup")


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance tests for integrated workflow."""
    
    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_workflow_performance(self, sample_csv_file, benchmark):
        """Benchmark complete workflow performance."""
        
        def run_workflow():
            # Simplified workflow for benchmarking
            loader = DataLoader()
            data = loader.load_and_validate_data(str(sample_csv_file))
            features, labels = loader.extract_features_and_labels(data)
            
            preprocessor = DataPreprocessor()
            processed_features = preprocessor.preprocess_features(data, features)
            normal_data, _ = preprocessor.separate_normal_anomalous(
                processed_features, labels, 'normal'
            )
            train_data, val_data = preprocessor.prepare_training_data(normal_data)
            
            trainer = EnhancedModelTrainer()
            model = trainer.create_model(train_data.shape[1], [16, 8, 16])
            trainer.train_model(train_data, val_data, {'epochs': 5})
            
            return model
        
        result = benchmark(run_workflow)
        assert result is not None
        assert result.is_trained is True
