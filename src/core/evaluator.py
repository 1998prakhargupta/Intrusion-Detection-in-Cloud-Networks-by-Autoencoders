"""Model evaluation and threshold calculation module for NIDS."""

import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

from ..utils.logger import get_logger
from ..utils.constants import ThresholdDefaults, PerformanceConstants


class ThresholdCalculator:
    """Enhanced threshold calculation with multiple methods."""
    
    def __init__(self):
        """Initialize threshold calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_all_thresholds(self, normal_errors: np.ndarray, 
                                anomalous_errors: Optional[np.ndarray] = None,
                                percentile: float = 95.0) -> Dict[str, float]:
        """Calculate multiple thresholds using different methods.
        
        Args:
            normal_errors: Reconstruction errors for normal data.
            anomalous_errors: Reconstruction errors for anomalous data (optional).
            percentile: Percentile for threshold calculation.
            
        Returns:
            Dictionary containing different threshold values.
        """
        thresholds = {}
        
        # Percentile-based threshold
        thresholds['percentile'] = self.percentile_threshold(normal_errors, percentile)
        
        # Statistical threshold (mean + 2*std)
        thresholds['statistical'] = self.statistical_threshold(normal_errors)
        
        # IQR-based threshold
        thresholds['iqr'] = self.iqr_threshold(normal_errors)
        
        # If anomalous data is available, calculate optimal thresholds
        if anomalous_errors is not None:
            thresholds['roc_optimal'] = self.roc_optimal_threshold(normal_errors, anomalous_errors)
            thresholds['youden'] = self.youden_threshold(normal_errors, anomalous_errors)
        
        self.logger.info("Threshold calculation completed:")
        for method, threshold in thresholds.items():
            self.logger.info(f"  {method}: {threshold:.6f}")
        
        return thresholds
    
    def percentile_threshold(self, errors: np.ndarray, percentile: float = 95.0) -> float:
        """Calculate percentile-based threshold.
        
        Args:
            errors: Reconstruction errors.
            percentile: Percentile value.
            
        Returns:
            Threshold value.
        """
        return np.percentile(errors, percentile)
    
    def statistical_threshold(self, errors: np.ndarray, n_std: float = 2.0) -> float:
        """Calculate statistical threshold (mean + n*std).
        
        Args:
            errors: Reconstruction errors.
            n_std: Number of standard deviations.
            
        Returns:
            Threshold value.
        """
        return np.mean(errors) + n_std * np.std(errors)
    
    def iqr_threshold(self, errors: np.ndarray, multiplier: float = 1.5) -> float:
        """Calculate IQR-based threshold.
        
        Args:
            errors: Reconstruction errors.
            multiplier: IQR multiplier.
            
        Returns:
            Threshold value.
        """
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = q3 - q1
        return q3 + multiplier * iqr
    
    def roc_optimal_threshold(self, normal_errors: np.ndarray, anomalous_errors: np.ndarray) -> float:
        """Calculate ROC-optimal threshold.
        
        Args:
            normal_errors: Normal reconstruction errors.
            anomalous_errors: Anomalous reconstruction errors.
            
        Returns:
            Optimal threshold value.
        """
        # Combine errors and create labels
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        true_labels = np.concatenate([
            np.zeros(len(normal_errors)),
            np.ones(len(anomalous_errors))
        ])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, all_errors)
        
        # Find optimal threshold (maximize TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    
    def youden_threshold(self, normal_errors: np.ndarray, anomalous_errors: np.ndarray) -> float:
        """Calculate Youden's J statistic optimal threshold.
        
        Args:
            normal_errors: Normal reconstruction errors.
            anomalous_errors: Anomalous reconstruction errors.
            
        Returns:
            Youden optimal threshold.
        """
        # Combine errors and create labels
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        true_labels = np.concatenate([
            np.zeros(len(normal_errors)),
            np.ones(len(anomalous_errors))
        ])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, all_errors)
        
        # Find Youden's J optimal threshold (maximize sensitivity + specificity - 1)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]


class MetricsCalculator:
    """Enhanced metrics calculation for model evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Calculate confusion matrix components.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary with confusion matrix components.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    
    def calculate_classification_metrics(self, confusion: Dict[str, int]) -> Dict[str, float]:
        """Calculate classification metrics from confusion matrix.
        
        Args:
            confusion: Confusion matrix dictionary.
            
        Returns:
            Dictionary with classification metrics.
        """
        tp, tn, fp, fn = confusion['tp'], confusion['tn'], confusion['fp'], confusion['fn']
        
        # Calculate metrics with zero division handling
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False positive and negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        }
    
    def calculate_auc(self, normal_errors: np.ndarray, anomalous_errors: np.ndarray) -> float:
        """Calculate ROC-AUC score.
        
        Args:
            normal_errors: Normal reconstruction errors.
            anomalous_errors: Anomalous reconstruction errors.
            
        Returns:
            ROC-AUC score.
        """
        try:
            all_errors = np.concatenate([normal_errors, anomalous_errors])
            true_labels = np.concatenate([
                np.zeros(len(normal_errors)),
                np.ones(len(anomalous_errors))
            ])
            return roc_auc_score(true_labels, all_errors)
        except Exception as e:
            self.logger.error(f"Failed to calculate AUC: {e}")
            return 0.0
    
    def interpret_performance(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Interpret performance metrics with qualitative descriptions.
        
        Args:
            metrics: Performance metrics dictionary.
            
        Returns:
            Dictionary with performance interpretations.
        """
        interpretations = {}
        
        for metric, value in metrics.items():
            if metric == 'false_positive_rate':
                # Lower is better for FPR
                if value <= PerformanceConstants.EXCELLENT_FPR:
                    interpretations[metric] = "Excellent"
                elif value <= PerformanceConstants.GOOD_FPR:
                    interpretations[metric] = "Good"
                elif value <= 0.2:
                    interpretations[metric] = "Moderate"
                else:
                    interpretations[metric] = "Poor"
            else:
                # Higher is better for most metrics
                if value >= PerformanceConstants.EXCELLENT_AUC:
                    interpretations[metric] = "Excellent"
                elif value >= PerformanceConstants.GOOD_AUC:
                    interpretations[metric] = "Good"
                elif value >= PerformanceConstants.FAIR_AUC:
                    interpretations[metric] = "Moderate"
                else:
                    interpretations[metric] = "Poor"
        
        return interpretations


class ModelEvaluator:
    """Comprehensive model evaluator for NIDS autoencoder."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.logger = get_logger(__name__)
        self.threshold_calc = ThresholdCalculator()
        self.metrics_calc = MetricsCalculator()
        self.evaluation_results = {}
    
    def evaluate_model(self, model, normal_data: np.ndarray, anomalous_data: Optional[np.ndarray] = None,
                      class_info: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained autoencoder model.
            normal_data: Normal validation data.
            anomalous_data: Anomalous test data (optional).
            class_info: Class information for detailed analysis.
            
        Returns:
            Complete evaluation results.
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Calculate reconstruction errors
        normal_errors = model.reconstruction_error(normal_data)
        self.logger.info(f"Normal validation errors calculated: {len(normal_errors)} samples")
        self.logger.info(f"  Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
        
        results = {
            'normal_errors': {
                'mean': np.mean(normal_errors),
                'std': np.std(normal_errors),
                'min': np.min(normal_errors),
                'max': np.max(normal_errors),
                'median': np.median(normal_errors)
            }
        }
        
        # Calculate thresholds
        if anomalous_data is not None:
            anomalous_errors = model.reconstruction_error(anomalous_data)
            self.logger.info(f"Anomalous errors calculated: {len(anomalous_errors)} samples")
            self.logger.info(f"  Mean: {np.mean(anomalous_errors):.6f}, Std: {np.std(anomalous_errors):.6f}")
            
            results['anomalous_errors'] = {
                'mean': np.mean(anomalous_errors),
                'std': np.std(anomalous_errors),
                'min': np.min(anomalous_errors),
                'max': np.max(anomalous_errors),
                'median': np.median(anomalous_errors)
            }
            
            # Calculate thresholds
            thresholds = self.threshold_calc.calculate_all_thresholds(normal_errors, anomalous_errors)
            results['thresholds'] = thresholds
            
            # Evaluate performance for each threshold
            results['performance'] = self._evaluate_thresholds(
                normal_errors, anomalous_errors, thresholds
            )
            
            # Calculate AUC
            auc_score = self.metrics_calc.calculate_auc(normal_errors, anomalous_errors)
            results['roc_auc'] = auc_score
            self.logger.info(f"ROC-AUC Score: {auc_score:.3f}")
            
        else:
            # Unsupervised evaluation
            thresholds = {
                'percentile': self.threshold_calc.percentile_threshold(normal_errors),
                'statistical': self.threshold_calc.statistical_threshold(normal_errors),
                'iqr': self.threshold_calc.iqr_threshold(normal_errors)
            }
            results['thresholds'] = thresholds
            self.logger.info("Unsupervised thresholds calculated")
        
        # Class-wise analysis if available
        if class_info is not None and anomalous_data is not None:
            results['class_analysis'] = self._analyze_by_class(
                model, normal_data, anomalous_data, class_info, thresholds
            )
        
        self.evaluation_results = results
        self.logger.info("Model evaluation completed successfully!")
        
        return results
    
    def _evaluate_thresholds(self, normal_errors: np.ndarray, anomalous_errors: np.ndarray,
                           thresholds: Dict[str, float]) -> Dict[str, Dict]:
        """Evaluate performance for different thresholds.
        
        Args:
            normal_errors: Normal reconstruction errors.
            anomalous_errors: Anomalous reconstruction errors.
            thresholds: Dictionary of threshold values.
            
        Returns:
            Performance results for each threshold.
        """
        performance_results = {}
        
        for method, threshold in thresholds.items():
            # Create predictions
            all_errors = np.concatenate([normal_errors, anomalous_errors])
            true_labels = np.concatenate([
                np.zeros(len(normal_errors)),
                np.ones(len(anomalous_errors))
            ])
            predicted_labels = (all_errors > threshold).astype(int)
            
            # Calculate metrics
            confusion = self.metrics_calc.calculate_confusion_matrix(true_labels, predicted_labels)
            metrics = self.metrics_calc.calculate_classification_metrics(confusion)
            interpretations = self.metrics_calc.interpret_performance(metrics)
            
            performance_results[method] = {
                'threshold': threshold,
                'metrics': metrics,
                'confusion_matrix': confusion,
                'interpretations': interpretations
            }
            
            self.logger.info(f"{method.upper()} Performance:")
            self.logger.info(f"  Threshold: {threshold:.6f}")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            self.logger.info(f"  Precision: {metrics['precision']:.3f}")
            self.logger.info(f"  Recall: {metrics['recall']:.3f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        
        return performance_results
    
    def _analyze_by_class(self, model, normal_data: np.ndarray, anomalous_data: np.ndarray,
                         class_info: pd.Series, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance by traffic class.
        
        Args:
            model: Trained model.
            normal_data: Normal data.
            anomalous_data: Anomalous data.
            class_info: Class information.
            thresholds: Threshold values.
            
        Returns:
            Class-wise analysis results.
        """
        self.logger.info("Performing class-wise analysis...")
        
        # This is a simplified implementation
        # In practice, you'd need to align data with class_info
        unique_classes = class_info.unique()
        normal_class = [cls for cls in unique_classes if str(cls).lower() == 'normal'][0]
        
        # Use best performing threshold (e.g., Youden or ROC optimal)
        best_threshold = thresholds.get('youden', thresholds.get('roc_optimal', thresholds['percentile']))
        
        class_analysis = {
            'normal_class': normal_class,
            'anomaly_classes': [cls for cls in unique_classes if cls != normal_class],
            'best_threshold': best_threshold,
            'total_classes': len(unique_classes)
        }
        
        return class_analysis
    
    def get_best_threshold_method(self) -> Tuple[str, float]:
        """Get the best performing threshold method.
        
        Returns:
            Tuple of (method_name, threshold_value).
        """
        if 'performance' not in self.evaluation_results:
            raise ValueError("No performance results available. Run evaluation first.")
        
        # Find method with highest F1-score
        best_method = max(
            self.evaluation_results['performance'].keys(),
            key=lambda x: self.evaluation_results['performance'][x]['metrics']['f1_score']
        )
        
        best_threshold = self.evaluation_results['performance'][best_method]['threshold']
        
        self.logger.info(f"Best performing method: {best_method} (threshold: {best_threshold:.6f})")
        
        return best_method, best_threshold
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report string.
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report_lines = ["="*60, "NIDS AUTOENCODER EVALUATION REPORT", "="*60]
        
        # Model performance summary
        if 'roc_auc' in self.evaluation_results:
            auc_score = self.evaluation_results['roc_auc']
            report_lines.append(f"ROC-AUC Score: {auc_score:.3f}")
        
        # Threshold analysis
        if 'thresholds' in self.evaluation_results:
            report_lines.append("\nThreshold Values:")
            for method, threshold in self.evaluation_results['thresholds'].items():
                report_lines.append(f"  {method}: {threshold:.6f}")
        
        # Performance metrics
        if 'performance' in self.evaluation_results:
            report_lines.append("\nPerformance by Threshold Method:")
            for method, perf in self.evaluation_results['performance'].items():
                metrics = perf['metrics']
                report_lines.append(f"\n{method.upper()}:")
                report_lines.append(f"  Accuracy:  {metrics['accuracy']:.3f}")
                report_lines.append(f"  Precision: {metrics['precision']:.3f}")
                report_lines.append(f"  Recall:    {metrics['recall']:.3f}")
                report_lines.append(f"  F1-Score:  {metrics['f1_score']:.3f}")
        
        report_lines.append("="*60)
        
        return "\n".join(report_lines)
