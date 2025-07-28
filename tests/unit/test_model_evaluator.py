"""
Unit tests for ModelEvaluator module.

Tests the model evaluation, threshold calculation, and metrics generation functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.evaluator import ModelEvaluator
from utils.constants import ThresholdDefaults, ThresholdMethods


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        
        assert evaluator.evaluation_results == {}
        assert evaluator.thresholds == {}
        assert evaluator.performance_metrics == {}
        assert hasattr(evaluator, 'logger')
    
    @pytest.mark.unit
    def test_evaluate_model_success(self, mock_evaluation_results):
        """Test successful model evaluation."""
        evaluator = ModelEvaluator()
        
        # Mock model
        mock_model = Mock()
        mock_model.reconstruction_error.return_value = mock_evaluation_results['normal_errors']['values']
        mock_model.is_trained = True
        
        # Mock data
        normal_data = np.random.randn(100, 10)
        anomalous_data = np.random.randn(20, 10)
        class_info = pd.Series(['normal'] * 100 + ['attack'] * 20)
        
        results = evaluator.evaluate_model(
            model=mock_model,
            normal_data=normal_data,
            anomalous_data=anomalous_data,
            class_info=class_info
        )
        
        assert isinstance(results, dict)
        assert 'normal_errors' in results
        assert 'anomalous_errors' in results
        assert 'thresholds' in results
        assert 'performance' in results
    
    @pytest.mark.unit
    def test_evaluate_model_no_anomalous_data(self):
        """Test evaluation with only normal data."""
        evaluator = ModelEvaluator()
        
        # Mock model
        mock_model = Mock()
        mock_model.reconstruction_error.return_value = np.random.exponential(0.05, 100)
        mock_model.is_trained = True
        
        normal_data = np.random.randn(100, 10)
        
        results = evaluator.evaluate_model(
            model=mock_model,
            normal_data=normal_data,
            anomalous_data=None,
            class_info=None
        )
        
        assert 'normal_errors' in results
        assert 'thresholds' in results
        # Should not have anomalous_errors or roc_auc without anomalous data
        assert 'anomalous_errors' not in results or results['anomalous_errors'] is None
    
    @pytest.mark.unit
    def test_evaluate_model_untrained_model(self):
        """Test evaluation fails with untrained model."""
        evaluator = ModelEvaluator()
        
        mock_model = Mock()
        mock_model.is_trained = False
        
        normal_data = np.random.randn(100, 10)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            evaluator.evaluate_model(mock_model, normal_data)
    
    @pytest.mark.unit
    def test_calculate_reconstruction_errors(self):
        """Test reconstruction error calculation."""
        evaluator = ModelEvaluator()
        
        # Mock model
        mock_model = Mock()
        expected_errors = np.array([0.1, 0.2, 0.05, 0.15, 0.3])
        mock_model.reconstruction_error.return_value = expected_errors
        
        data = np.random.randn(5, 10)
        errors = evaluator._calculate_reconstruction_errors(mock_model, data)
        
        np.testing.assert_array_equal(errors, expected_errors)
        mock_model.reconstruction_error.assert_called_once_with(data)
    
    @pytest.mark.unit
    def test_calculate_error_statistics(self):
        """Test error statistics calculation."""
        evaluator = ModelEvaluator()
        
        errors = np.array([0.1, 0.2, 0.05, 0.15, 0.3, 0.08, 0.25])
        stats = evaluator._calculate_error_statistics(errors)
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'values' in stats
        
        assert stats['mean'] == np.mean(errors)
        assert stats['std'] == np.std(errors)
        assert stats['min'] == np.min(errors)
        assert stats['max'] == np.max(errors)
        assert stats['median'] == np.median(errors)
        np.testing.assert_array_equal(stats['values'], errors)
    
    @pytest.mark.unit
    def test_calculate_thresholds_percentile(self):
        """Test percentile threshold calculation."""
        evaluator = ModelEvaluator()
        
        normal_errors = np.array([0.1, 0.2, 0.05, 0.15, 0.3, 0.08, 0.25, 0.12, 0.18, 0.22])
        thresholds = evaluator._calculate_thresholds(normal_errors)
        
        assert 'percentile' in thresholds
        expected_percentile = np.percentile(normal_errors, ThresholdDefaults.PERCENTILE_VALUE)
        assert abs(thresholds['percentile'] - expected_percentile) < 1e-10
    
    @pytest.mark.unit
    def test_calculate_thresholds_statistical(self):
        """Test statistical threshold calculation."""
        evaluator = ModelEvaluator()
        
        normal_errors = np.array([0.1, 0.2, 0.05, 0.15, 0.3, 0.08, 0.25, 0.12, 0.18, 0.22])
        thresholds = evaluator._calculate_thresholds(normal_errors)
        
        assert 'statistical' in thresholds
        expected_statistical = np.mean(normal_errors) + ThresholdDefaults.STATISTICAL_N_STD * np.std(normal_errors)
        assert abs(thresholds['statistical'] - expected_statistical) < 1e-10
    
    @pytest.mark.unit
    def test_calculate_thresholds_roc_optimal(self):
        """Test ROC optimal threshold calculation."""
        evaluator = ModelEvaluator()
        
        normal_errors = np.random.exponential(0.1, 100)
        anomalous_errors = np.random.exponential(0.3, 50)
        
        thresholds = evaluator._calculate_thresholds(normal_errors, anomalous_errors)
        
        assert 'roc_optimal' in thresholds
        assert thresholds['roc_optimal'] > 0
    
    @pytest.mark.unit
    def test_calculate_performance_metrics_binary(self):
        """Test binary classification performance metrics."""
        evaluator = ModelEvaluator()
        
        # Create test data
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])  # 0=normal, 1=anomaly
        y_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.7, 0.25, 0.85, 0.12, 0.75])
        threshold = 0.5
        
        metrics = evaluator._calculate_performance_metrics(y_true, y_scores, threshold)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check values are in valid ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    @pytest.mark.unit
    def test_calculate_roc_auc(self):
        """Test ROC-AUC calculation."""
        evaluator = ModelEvaluator()
        
        # Create test data with clear separation
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.7, 0.8, 0.9, 0.85, 0.95])
        
        auc_score = evaluator._calculate_roc_auc(y_true, y_scores)
        
        assert isinstance(auc_score, float)
        assert 0 <= auc_score <= 1
        # With clear separation, AUC should be high
        assert auc_score > 0.8
    
    @pytest.mark.unit
    def test_calculate_roc_auc_perfect_separation(self):
        """Test ROC-AUC with perfect separation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        auc_score = evaluator._calculate_roc_auc(y_true, y_scores)
        
        # Should be perfect or near-perfect
        assert auc_score >= 0.95
    
    @pytest.mark.unit
    def test_calculate_roc_auc_random_classification(self):
        """Test ROC-AUC with random classification."""
        evaluator = ModelEvaluator()
        
        # Random labels and scores
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_scores = np.random.random(100)
        
        auc_score = evaluator._calculate_roc_auc(y_true, y_scores)
        
        # Should be around 0.5 for random classification
        assert 0.3 <= auc_score <= 0.7
    
    @pytest.mark.unit
    def test_get_best_threshold_method(self, mock_evaluation_results):
        """Test getting the best threshold method."""
        evaluator = ModelEvaluator()
        evaluator.evaluation_results = mock_evaluation_results
        
        best_method, best_threshold = evaluator.get_best_threshold_method()
        
        assert best_method in ['percentile', 'statistical', 'roc_optimal']
        assert isinstance(best_threshold, float)
        assert best_threshold > 0
    
    @pytest.mark.unit
    def test_get_best_threshold_method_no_results(self):
        """Test getting best threshold method when no results available."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="No evaluation results"):
            evaluator.get_best_threshold_method()
    
    @pytest.mark.unit
    def test_generate_evaluation_report(self, mock_evaluation_results):
        """Test evaluation report generation."""
        evaluator = ModelEvaluator()
        evaluator.evaluation_results = mock_evaluation_results
        
        report = evaluator.generate_evaluation_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'recommendations' in report
        assert 'model_performance' in report
        
        assert 'overall_score' in report['summary']
        assert 'best_threshold_method' in report['summary']
    
    @pytest.mark.unit
    def test_generate_evaluation_report_no_results(self):
        """Test report generation when no results available."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="No evaluation results"):
            evaluator.generate_evaluation_report()
    
    @pytest.mark.unit
    def test_save_evaluation_results(self, mock_evaluation_results, test_data_dir):
        """Test saving evaluation results."""
        evaluator = ModelEvaluator()
        evaluator.evaluation_results = mock_evaluation_results
        
        save_path = test_data_dir / "evaluation_results.pkl"
        evaluator.save_evaluation_results(str(save_path))
        
        assert save_path.exists()
    
    @pytest.mark.unit
    def test_load_evaluation_results(self, mock_evaluation_results, test_data_dir):
        """Test loading evaluation results."""
        # First save results
        evaluator1 = ModelEvaluator()
        evaluator1.evaluation_results = mock_evaluation_results
        
        save_path = test_data_dir / "evaluation_results.pkl"
        evaluator1.save_evaluation_results(str(save_path))
        
        # Now load in new evaluator
        evaluator2 = ModelEvaluator()
        evaluator2.load_evaluation_results(str(save_path))
        
        assert evaluator2.evaluation_results == mock_evaluation_results
    
    @pytest.mark.unit
    def test_performance_metrics_edge_cases(self):
        """Test performance metrics with edge cases."""
        evaluator = ModelEvaluator()
        
        # All predictions correct
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = 0.5
        
        metrics = evaluator._calculate_performance_metrics(y_true, y_scores, threshold)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    @pytest.mark.unit
    def test_threshold_methods_parameter(self, threshold_methods):
        """Test different threshold calculation methods."""
        evaluator = ModelEvaluator()
        
        normal_errors = np.random.exponential(0.1, 100)
        anomalous_errors = np.random.exponential(0.3, 50)
        
        if threshold_methods == 'roc_optimal':
            thresholds = evaluator._calculate_thresholds(normal_errors, anomalous_errors)
        else:
            thresholds = evaluator._calculate_thresholds(normal_errors)
        
        assert threshold_methods in thresholds
        assert thresholds[threshold_methods] > 0
    
    @pytest.mark.unit
    def test_empty_data_handling(self):
        """Test handling of empty data arrays."""
        evaluator = ModelEvaluator()
        
        empty_errors = np.array([])
        
        with pytest.raises(Exception):  # Should raise some error
            evaluator._calculate_error_statistics(empty_errors)
    
    @pytest.mark.unit
    def test_single_value_arrays(self):
        """Test handling of single-value arrays."""
        evaluator = ModelEvaluator()
        
        single_error = np.array([0.5])
        stats = evaluator._calculate_error_statistics(single_error)
        
        assert stats['mean'] == 0.5
        assert stats['std'] == 0.0  # Standard deviation of single value
        assert stats['min'] == 0.5
        assert stats['max'] == 0.5
    
    @pytest.mark.unit
    def test_identical_values_arrays(self):
        """Test handling of arrays with identical values."""
        evaluator = ModelEvaluator()
        
        identical_errors = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        stats = evaluator._calculate_error_statistics(identical_errors)
        
        assert stats['mean'] == 0.5
        assert stats['std'] == 0.0
        assert stats['min'] == 0.5
        assert stats['max'] == 0.5
    
    @pytest.mark.unit
    def test_very_large_values(self):
        """Test handling of very large error values."""
        evaluator = ModelEvaluator()
        
        large_errors = np.array([1e6, 1e7, 1e8])
        stats = evaluator._calculate_error_statistics(large_errors)
        
        assert stats['mean'] > 1e6
        assert stats['std'] > 0
        assert stats['min'] == 1e6
        assert stats['max'] == 1e8
    
    @pytest.mark.unit
    def test_negative_error_values(self):
        """Test handling of negative error values (should not occur but test robustness)."""
        evaluator = ModelEvaluator()
        
        # This shouldn't happen in practice but test robustness
        mixed_errors = np.array([-0.1, 0.1, -0.2, 0.3, 0.5])
        stats = evaluator._calculate_error_statistics(mixed_errors)
        
        assert stats['mean'] == np.mean(mixed_errors)
        assert stats['min'] == -0.2
        assert stats['max'] == 0.5


class TestModelEvaluatorIntegration:
    """Integration tests for ModelEvaluator with complete workflows."""
    
    @pytest.mark.integration
    def test_complete_evaluation_workflow(self, trained_model_data, normal_data_sample, anomalous_data_sample):
        """Test complete evaluation workflow."""
        evaluator = ModelEvaluator()
        
        # Mock a trained model
        mock_model = Mock()
        mock_model.is_trained = True
        mock_model.reconstruction_error.side_effect = lambda x: np.random.exponential(
            0.1 if len(x) == len(normal_data_sample) else 0.3, len(x)
        )
        
        # Run complete evaluation
        results = evaluator.evaluate_model(
            model=mock_model,
            normal_data=normal_data_sample,
            anomalous_data=anomalous_data_sample,
            class_info=None
        )
        
        # Verify complete results
        assert 'normal_errors' in results
        assert 'anomalous_errors' in results
        assert 'thresholds' in results
        assert 'performance' in results
        
        # Get best threshold
        best_method, best_threshold = evaluator.get_best_threshold_method()
        assert best_threshold > 0
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        assert 'summary' in report
    
    @pytest.mark.integration
    def test_evaluation_with_real_autoencoder(self, normal_data_sample):
        """Test evaluation with actual ProductionAutoencoder."""
        # This test would require importing ProductionAutoencoder
        # For now, we'll use a mock but this shows the integration pattern
        evaluator = ModelEvaluator()
        
        # Create mock autoencoder that behaves realistically
        mock_autoencoder = Mock()
        mock_autoencoder.is_trained = True
        
        def realistic_reconstruction_error(data):
            # Simulate realistic reconstruction errors
            # Normal data should have lower errors
            return np.random.exponential(0.05, len(data))
        
        mock_autoencoder.reconstruction_error = realistic_reconstruction_error
        
        results = evaluator.evaluate_model(
            model=mock_autoencoder,
            normal_data=normal_data_sample,
            anomalous_data=None
        )
        
        assert results['normal_errors']['mean'] > 0
        assert 'thresholds' in results
    
    @pytest.mark.integration
    def test_evaluation_persistence_workflow(self, mock_evaluation_results, test_data_dir):
        """Test complete evaluation persistence workflow."""
        evaluator1 = ModelEvaluator()
        evaluator1.evaluation_results = mock_evaluation_results
        
        # Save results
        save_path = test_data_dir / "evaluation.pkl"
        evaluator1.save_evaluation_results(str(save_path))
        
        # Generate and save report
        report = evaluator1.generate_evaluation_report()
        assert 'summary' in report
        
        # Load results in new evaluator
        evaluator2 = ModelEvaluator()
        evaluator2.load_evaluation_results(str(save_path))
        
        # Should be able to generate same report
        report2 = evaluator2.generate_evaluation_report()
        assert report2['summary']['overall_score'] == report['summary']['overall_score']
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_evaluation_performance_large_dataset(self, large_dataset):
        """Test evaluation performance with large datasets."""
        import time
        
        evaluator = ModelEvaluator()
        
        # Create large mock data
        normal_data = np.random.randn(5000, 10)
        anomalous_data = np.random.randn(1000, 10)
        
        # Mock model with realistic behavior
        mock_model = Mock()
        mock_model.is_trained = True
        
        def mock_reconstruction_error(data):
            if len(data) == 5000:  # Normal data
                return np.random.exponential(0.1, len(data))
            else:  # Anomalous data
                return np.random.exponential(0.3, len(data))
        
        mock_model.reconstruction_error = mock_reconstruction_error
        
        start_time = time.time()
        
        results = evaluator.evaluate_model(
            model=mock_model,
            normal_data=normal_data,
            anomalous_data=anomalous_data
        )
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds max
        assert 'roc_auc' in results
        assert results['roc_auc'] > 0.5  # Should be better than random
