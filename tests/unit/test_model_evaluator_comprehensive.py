"""
Comprehensive unit tests for ModelEvaluator module.

This module tests all aspects of model evaluation, threshold calculation, and metrics.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.evaluator import ModelEvaluator
from core.enhanced_trainer import ProductionAutoencoder


class TestModelEvaluator:
    """Comprehensive test suite for ModelEvaluator class."""
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        
        assert hasattr(evaluator, 'logger')
        assert evaluator.evaluation_results is None
        assert evaluator.best_threshold is None
        assert evaluator.best_method is None
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model_normal_only(self, normal_data_sample):
        """Test model evaluation with only normal data."""
        evaluator = ModelEvaluator()
        
        # Create and train a simple model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        results = evaluator.evaluate_model(
            model=model,
            normal_data=normal_data_sample,
            anomalous_data=None,
            class_info=None
        )
        
        assert isinstance(results, dict)
        assert 'normal_errors' in results
        assert 'thresholds' in results
        
        # Check normal error statistics
        normal_stats = results['normal_errors']
        assert 'mean' in normal_stats
        assert 'std' in normal_stats
        assert 'min' in normal_stats
        assert 'max' in normal_stats
        assert normal_stats['mean'] >= 0
        assert normal_stats['std'] >= 0
        assert normal_stats['min'] <= normal_stats['max']
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model_with_anomalies(self, normal_data_sample, anomalous_data_sample):
        """Test model evaluation with both normal and anomalous data."""
        evaluator = ModelEvaluator()
        
        # Create and train model on normal data
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        # Create class info
        normal_labels = ['normal'] * len(normal_data_sample)
        anomalous_labels = ['attack'] * len(anomalous_data_sample)
        class_info = pd.Series(normal_labels + anomalous_labels)
        
        # Combine data for evaluation
        all_data = np.vstack([normal_data_sample, anomalous_data_sample])
        
        results = evaluator.evaluate_model(
            model=model,
            normal_data=normal_data_sample,
            anomalous_data=anomalous_data_sample,
            class_info=class_info
        )
        
        assert 'normal_errors' in results
        assert 'anomalous_errors' in results
        assert 'roc_auc' in results
        assert 'performance' in results
        
        # Check that anomalous errors are generally higher
        normal_mean = results['normal_errors']['mean']
        anomalous_mean = results['anomalous_errors']['mean']
        assert anomalous_mean > normal_mean
        
        # ROC-AUC should be between 0 and 1
        assert 0 <= results['roc_auc'] <= 1
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_calculate_thresholds(self, normal_data_sample):
        """Test threshold calculation methods."""
        evaluator = ModelEvaluator()
        
        # Create reconstruction errors
        errors = np.random.exponential(1.0, 1000)  # Simulated reconstruction errors
        
        thresholds = evaluator.calculate_thresholds(errors)
        
        assert isinstance(thresholds, dict)
        assert 'percentile' in thresholds
        assert 'statistical' in thresholds
        
        # Check threshold values are reasonable
        assert thresholds['percentile'] > 0
        assert thresholds['statistical'] > 0
        
        # Percentile threshold should be around 95th percentile
        expected_percentile = np.percentile(errors, 95)
        assert abs(thresholds['percentile'] - expected_percentile) < 0.1
        
        # Statistical threshold should be mean + n*std
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        expected_statistical = mean_error + 2 * std_error
        assert abs(thresholds['statistical'] - expected_statistical) < 0.1
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_calculate_metrics(self):
        """Test performance metrics calculation."""
        evaluator = ModelEvaluator()
        
        # Create mock predictions and ground truth
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])  # 0=normal, 1=anomaly
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])  # Predictions
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'false_positive_rate' in metrics
        
        # Check metric ranges
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1"
        
        # Manually verify accuracy
        expected_accuracy = np.sum(y_true == y_pred) / len(y_true)
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.001
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_get_best_threshold_method(self, normal_data_sample, anomalous_data_sample):
        """Test getting best threshold method."""
        evaluator = ModelEvaluator()
        
        # Create and train model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        # Create class info
        normal_labels = ['normal'] * len(normal_data_sample)
        anomalous_labels = ['attack'] * len(anomalous_data_sample)
        class_info = pd.Series(normal_labels + anomalous_labels)
        
        # Evaluate model
        evaluator.evaluate_model(
            model=model,
            normal_data=normal_data_sample,
            anomalous_data=anomalous_data_sample,
            class_info=class_info
        )
        
        best_method, best_threshold = evaluator.get_best_threshold_method()
        
        assert best_method is not None
        assert best_threshold is not None
        assert isinstance(best_method, str)
        assert isinstance(best_threshold, (int, float))
        assert best_threshold > 0
        
        # Should be one of the calculated methods
        assert best_method in ['percentile', 'statistical']
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_generate_evaluation_report(self, normal_data_sample, anomalous_data_sample):
        """Test evaluation report generation."""
        evaluator = ModelEvaluator()
        
        # Create and train model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        # Evaluate model
        evaluator.evaluate_model(
            model=model,
            normal_data=normal_data_sample,
            anomalous_data=anomalous_data_sample,
            class_info=None
        )
        
        report = evaluator.generate_evaluation_report()
        
        assert isinstance(report, dict)
        assert 'model_performance' in report
        assert 'threshold_analysis' in report
        assert 'recommendations' in report
        assert 'summary_statistics' in report
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_different_threshold_methods(self, normal_data_sample):
        """Test evaluation with different threshold methods."""
        evaluator = ModelEvaluator()
        
        # Create reconstruction errors with known distribution
        errors = np.concatenate([
            np.random.normal(1.0, 0.5, 800),  # Normal errors
            np.random.normal(3.0, 1.0, 200)   # Anomalous errors
        ])
        
        # Test percentile threshold
        percentile_threshold = evaluator._calculate_percentile_threshold(errors, percentile=95)
        assert percentile_threshold > 0
        
        # Test statistical threshold
        statistical_threshold = evaluator._calculate_statistical_threshold(errors, n_std=2)
        assert statistical_threshold > 0
        
        # They should be different values
        assert abs(percentile_threshold - statistical_threshold) > 0.1
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_roc_auc_calculation(self):
        """Test ROC-AUC calculation."""
        evaluator = ModelEvaluator()
        
        # Create perfect classifier case
        y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
        scores_perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        
        auc_perfect = evaluator._calculate_roc_auc(y_true_perfect, scores_perfect)
        assert auc_perfect == 1.0
        
        # Create random classifier case
        y_true_random = np.array([0, 1, 0, 1, 0, 1])
        scores_random = np.array([0.4, 0.6, 0.5, 0.5, 0.6, 0.4])
        
        auc_random = evaluator._calculate_roc_auc(y_true_random, scores_random)
        assert 0.4 <= auc_random <= 0.6  # Should be around 0.5 for random
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_performance_by_threshold(self, normal_data_sample, anomalous_data_sample):
        """Test performance calculation by different thresholds."""
        evaluator = ModelEvaluator()
        
        # Create and train model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        # Get reconstruction errors
        normal_errors = model.reconstruction_error(normal_data_sample)
        anomalous_errors = model.reconstruction_error(anomalous_data_sample)
        
        # Calculate thresholds
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        thresholds = evaluator.calculate_thresholds(all_errors)
        
        # Create ground truth
        y_true = np.concatenate([
            np.zeros(len(normal_errors)),
            np.ones(len(anomalous_errors))
        ])
        
        performance = evaluator._calculate_performance_by_threshold(
            normal_errors, anomalous_errors, y_true, thresholds
        )
        
        assert isinstance(performance, dict)
        
        for method in thresholds.keys():
            assert method in performance
            assert 'threshold' in performance[method]
            assert 'metrics' in performance[method]
            
            metrics = performance[method]['metrics']
            assert all(0 <= metrics[m] <= 1 for m in metrics.values())


class TestModelEvaluatorErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model_no_model(self, normal_data_sample):
        """Test evaluation without model."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="Model is required"):
            evaluator.evaluate_model(
                model=None,
                normal_data=normal_data_sample
            )
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_model_no_data(self):
        """Test evaluation without data."""
        evaluator = ModelEvaluator()
        
        # Create dummy model
        model = ProductionAutoencoder(10, [8, 6, 8])
        
        with pytest.raises(ValueError, match="Normal data is required"):
            evaluator.evaluate_model(
                model=model,
                normal_data=None
            )
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_evaluate_untrained_model(self, normal_data_sample):
        """Test evaluation of untrained model."""
        evaluator = ModelEvaluator()
        
        # Create untrained model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        
        with pytest.raises(ValueError, match="Model must be trained"):
            evaluator.evaluate_model(
                model=model,
                normal_data=normal_data_sample
            )
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_calculate_thresholds_empty_errors(self):
        """Test threshold calculation with empty errors."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="Empty reconstruction errors"):
            evaluator.calculate_thresholds(np.array([]))
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_calculate_metrics_inconsistent_lengths(self):
        """Test metrics calculation with inconsistent array lengths."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Different length
        
        with pytest.raises(ValueError):
            evaluator._calculate_metrics(y_true, y_pred)
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_get_best_threshold_without_evaluation(self):
        """Test getting best threshold without prior evaluation."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="No evaluation results available"):
            evaluator.get_best_threshold_method()
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_roc_auc_single_class(self):
        """Test ROC-AUC with single class (should handle gracefully)."""
        evaluator = ModelEvaluator()
        
        # All samples from one class
        y_true_single = np.array([0, 0, 0, 0, 0])
        scores_single = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Should handle gracefully (return NaN or specific value)
        try:
            auc = evaluator._calculate_roc_auc(y_true_single, scores_single)
            assert np.isnan(auc) or auc == 0.5
        except ValueError:
            # This is also acceptable behavior
            pass


class TestModelEvaluatorFallbackLogic:
    """Test fallback logic and production utilities."""
    
    @pytest.mark.unit
    @pytest.mark.model
    @patch('core.evaluator.get_logger')
    def test_logger_fallback(self, mock_get_logger):
        """Test logger initialization fallback."""
        mock_get_logger.side_effect = ImportError("Logger not available")
        
        with patch('logging.getLogger') as mock_logging:
            mock_logging.return_value = Mock()
            evaluator = ModelEvaluator()
            assert evaluator is not None
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_metrics_calculation_fallback(self):
        """Test fallback when sklearn metrics are not available."""
        evaluator = ModelEvaluator()
        
        # Test manual implementation of basic metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        # Should calculate metrics manually if sklearn not available
        with patch('core.evaluator.accuracy_score', side_effect=ImportError):
            try:
                metrics = evaluator._calculate_metrics(y_true, y_pred)
                assert 'accuracy' in metrics
            except ImportError:
                # Acceptable if no fallback implementation
                pass
    
    @pytest.mark.unit
    @pytest.mark.model
    def test_threshold_calculation_robust(self):
        """Test robust threshold calculation with edge cases."""
        evaluator = ModelEvaluator()
        
        # Test with very small errors
        small_errors = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        thresholds = evaluator.calculate_thresholds(small_errors)
        
        assert all(t > 0 for t in thresholds.values())
        
        # Test with very large errors
        large_errors = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        thresholds_large = evaluator.calculate_thresholds(large_errors)
        
        assert all(np.isfinite(t) for t in thresholds_large.values())


class TestModelEvaluatorIntegration:
    """Integration tests for ModelEvaluator."""
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_complete_evaluation_workflow(self, normal_data_sample, anomalous_data_sample):
        """Test complete evaluation workflow."""
        evaluator = ModelEvaluator()
        
        # 1. Create and train model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [16, 8, 16])
        model.train(normal_data_sample, epochs=10)
        
        # 2. Create comprehensive class info
        normal_labels = ['normal'] * len(normal_data_sample)
        attack_labels = ['attack'] * len(anomalous_data_sample)
        class_info = pd.Series(normal_labels + attack_labels)
        
        # 3. Perform evaluation
        results = evaluator.evaluate_model(
            model=model,
            normal_data=normal_data_sample,
            anomalous_data=anomalous_data_sample,
            class_info=class_info
        )
        
        # 4. Get best threshold
        best_method, best_threshold = evaluator.get_best_threshold_method()
        
        # 5. Generate report
        report = evaluator.generate_evaluation_report()
        
        # Verify complete workflow
        assert results is not None
        assert best_method is not None
        assert best_threshold is not None
        assert report is not None
        
        # Check quality metrics
        assert results['roc_auc'] > 0.5  # Should be better than random
        assert 'performance' in results
        assert len(results['performance']) >= 2  # At least 2 threshold methods
    
    @pytest.mark.integration
    @pytest.mark.model
    def test_evaluation_reproducibility(self, normal_data_sample, anomalous_data_sample):
        """Test that evaluation is reproducible."""
        # Run evaluation twice
        evaluator1 = ModelEvaluator()
        evaluator2 = ModelEvaluator()
        
        # Create identical models and train them
        np.random.seed(42)
        input_dim = normal_data_sample.shape[1]
        model1 = ProductionAutoencoder(input_dim, [8, 6, 8])
        model1.train(normal_data_sample, epochs=5)
        
        np.random.seed(42)
        model2 = ProductionAutoencoder(input_dim, [8, 6, 8])
        model2.train(normal_data_sample, epochs=5)
        
        # Evaluate both
        results1 = evaluator1.evaluate_model(model1, normal_data_sample, anomalous_data_sample)
        results2 = evaluator2.evaluate_model(model2, normal_data_sample, anomalous_data_sample)
        
        # Results should be very similar (allowing for small numerical differences)
        assert abs(results1['normal_errors']['mean'] - results2['normal_errors']['mean']) < 1e-6
        assert abs(results1['anomalous_errors']['mean'] - results2['anomalous_errors']['mean']) < 1e-6


@pytest.mark.slow
class TestModelEvaluatorPerformance:
    """Performance tests for ModelEvaluator."""
    
    @pytest.mark.benchmark
    def test_evaluation_performance(self, normal_data_sample, anomalous_data_sample, benchmark):
        """Benchmark evaluation performance."""
        evaluator = ModelEvaluator()
        
        # Create and train model
        input_dim = normal_data_sample.shape[1]
        model = ProductionAutoencoder(input_dim, [8, 6, 8])
        model.train(normal_data_sample, epochs=5)
        
        def run_evaluation():
            return evaluator.evaluate_model(
                model=model,
                normal_data=normal_data_sample,
                anomalous_data=anomalous_data_sample
            )
        
        result = benchmark(run_evaluation)
        assert result is not None
    
    @pytest.mark.memory
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        evaluator = ModelEvaluator()
        
        # Create large dataset
        n_normal = 5000
        n_anomalous = 1000
        n_features = 20
        
        large_normal = np.random.randn(n_normal, n_features)
        large_anomalous = np.random.randn(n_anomalous, n_features) + 2  # Shifted distribution
        
        # Create and train model
        model = ProductionAutoencoder(n_features, [16, 8, 16])
        model.train(large_normal[:1000], epochs=5)  # Train on subset
        
        # Evaluate on full dataset
        results = evaluator.evaluate_model(
            model=model,
            normal_data=large_normal,
            anomalous_data=large_anomalous
        )
        
        assert results is not None
        assert 'roc_auc' in results
        assert results['roc_auc'] > 0.5  # Should detect the shifted distribution
