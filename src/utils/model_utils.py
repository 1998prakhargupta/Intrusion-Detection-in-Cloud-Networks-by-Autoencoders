"""Model utilities for training, evaluation, and management."""

import time
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .constants import ModelDefaults, ThresholdMethods, PerformanceConstants, ErrorMessages
from .logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Utility class for model management operations."""
    
    @staticmethod
    def save_model_metadata(model_path: Path, metadata: Dict[str, Any]) -> None:
        """Save model metadata to JSON file.
        
        Args:
            model_path: Path to the model file.
            metadata: Metadata dictionary to save.
        """
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        
        # Add timestamp
        metadata['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metadata['model_file'] = str(model_path.name)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to: {metadata_path}")
    
    @staticmethod
    def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
        """Load model metadata from JSON file.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            Metadata dictionary or None if not found.
        """
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"No metadata found for model: {model_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from: {metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    @staticmethod
    def create_model_checkpoint(model, model_path: Path, 
                              metrics: Dict[str, float],
                              config: Dict[str, Any]) -> None:
        """Create a model checkpoint with metadata.
        
        Args:
            model: Model object to save.
            model_path: Path to save the model.
            metrics: Training metrics.
            config: Model configuration.
        """
        # Save model
        if TORCH_AVAILABLE and hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'metrics': metrics,
            'config': config,
            'model_type': type(model).__name__,
            'framework': 'pytorch' if TORCH_AVAILABLE and hasattr(model, 'state_dict') else 'numpy'
        }
        
        ModelManager.save_model_metadata(model_path, metadata)


class ThresholdCalculator:
    """Utility class for threshold calculations."""
    
    @staticmethod
    def percentile_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
        """Calculate percentile-based threshold."""
        return float(np.percentile(errors, percentile))
    
    @staticmethod
    def statistical_threshold(errors: np.ndarray, n_std: float = 2.0) -> float:
        """Calculate statistical threshold (mean + n*std)."""
        return float(np.mean(errors) + n_std * np.std(errors))
    
    @staticmethod
    def youden_threshold(normal_errors: np.ndarray, 
                        anomalous_errors: np.ndarray) -> float:
        """Calculate optimal threshold using Youden's J statistic.
        
        Returns:
            Optimal threshold value.
        """
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        all_labels = np.concatenate([
            np.zeros(len(normal_errors)),
            np.ones(len(anomalous_errors))
        ])
        
        # Sort by error values
        sorted_indices = np.argsort(all_errors)
        sorted_errors = all_errors[sorted_indices]
        
        best_threshold = 0.0
        best_j_score = -1.0
        
        # Sample thresholds for efficiency
        n_samples = min(PerformanceConstants.AUC_SAMPLE_POINTS, len(sorted_errors))
        sample_indices = np.linspace(0, len(sorted_errors) - 1, n_samples, dtype=int)
        
        for i in sample_indices:
            threshold = sorted_errors[i]
            predicted = (all_errors > threshold).astype(int)
            
            tp = np.sum((all_labels == 1) & (predicted == 1))
            fn = np.sum((all_labels == 1) & (predicted == 0))
            fp = np.sum((all_labels == 0) & (predicted == 1))
            tn = np.sum((all_labels == 0) & (predicted == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            j_score = tpr - fpr  # Youden's J statistic
            
            if j_score > best_j_score:
                best_j_score = j_score
                best_threshold = threshold
        
        return float(best_threshold)
    
    @staticmethod
    def calculate_all_thresholds(normal_errors: np.ndarray,
                               anomalous_errors: Optional[np.ndarray] = None,
                               percentile: float = ModelDefaults.THRESHOLD_PERCENTILE) -> Dict[str, float]:
        """Calculate all available threshold methods.
        
        Args:
            normal_errors: Reconstruction errors from normal data.
            anomalous_errors: Reconstruction errors from anomalous data (optional).
            percentile: Percentile value for percentile method.
            
        Returns:
            Dictionary of threshold methods and values.
        """
        thresholds = {}
        
        # Percentile threshold
        thresholds[ThresholdMethods.PERCENTILE.value] = ThresholdCalculator.percentile_threshold(
            normal_errors, percentile
        )
        
        # Statistical threshold
        thresholds[ThresholdMethods.STATISTICAL.value] = ThresholdCalculator.statistical_threshold(
            normal_errors
        )
        
        # ROC-optimal threshold (requires anomalous data)
        if anomalous_errors is not None:
            youden_threshold = ThresholdCalculator.youden_threshold(
                normal_errors, anomalous_errors
            )
            thresholds[ThresholdMethods.ROC_OPTIMAL.value] = youden_threshold
            thresholds[ThresholdMethods.YOUDEN.value] = youden_threshold
        
        return thresholds


class MetricsCalculator:
    """Utility class for performance metrics calculation."""
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Calculate confusion matrix components.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly).
            y_pred: Predicted labels (0=normal, 1=anomaly).
            
        Returns:
            Dictionary with tp, tn, fp, fn counts.
        """
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    
    @staticmethod
    def calculate_classification_metrics(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
        """Calculate classification metrics from confusion matrix.
        
        Args:
            confusion_matrix: Dictionary with tp, tn, fp, fn.
            
        Returns:
            Dictionary with calculated metrics.
        """
        tp, tn, fp, fn = confusion_matrix['tp'], confusion_matrix['tn'], confusion_matrix['fp'], confusion_matrix['fn']
        
        # Calculate metrics with safe division
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Additional metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1_score),
            'fpr': float(fpr),
            'fnr': float(fnr)
        }
    
    @staticmethod
    def calculate_auc(normal_errors: np.ndarray, anomalous_errors: np.ndarray) -> float:
        """Calculate AUC score using trapezoidal rule.
        
        Args:
            normal_errors: Reconstruction errors from normal data.
            anomalous_errors: Reconstruction errors from anomalous data.
            
        Returns:
            AUC score.
        """
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        all_labels = np.concatenate([
            np.zeros(len(normal_errors)),
            np.ones(len(anomalous_errors))
        ])
        
        # Sort by error values
        sorted_indices = np.argsort(all_errors)
        sorted_errors = all_errors[sorted_indices]
        sorted_labels = all_labels[sorted_indices]
        
        tpr_values = []
        fpr_values = []
        
        # Sample thresholds for efficiency
        n_samples = min(PerformanceConstants.AUC_SAMPLE_POINTS, len(sorted_errors))
        sample_indices = np.linspace(0, len(sorted_errors) - 1, n_samples, dtype=int)
        
        for i in sample_indices:
            threshold_idx = i
            predicted = np.zeros(len(sorted_labels))
            predicted[threshold_idx:] = 1
            
            confusion = MetricsCalculator.calculate_confusion_matrix(sorted_labels, predicted)
            metrics = MetricsCalculator.calculate_classification_metrics(confusion)
            
            tpr_values.append(metrics['recall'])
            fpr_values.append(metrics['fpr'])
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return abs(float(auc))
    
    @staticmethod
    def interpret_performance(metrics: Dict[str, float]) -> Dict[str, str]:
        """Interpret performance metrics and provide recommendations.
        
        Args:
            metrics: Dictionary with performance metrics.
            
        Returns:
            Dictionary with interpretations and recommendations.
        """
        interpretations = {}
        
        # AUC interpretation
        if 'auc' in metrics:
            interpretations['auc'] = MetricsCalculator._interpret_auc(metrics['auc'])
        
        # False Positive Rate interpretation
        if 'fpr' in metrics:
            interpretations['fpr'] = MetricsCalculator._interpret_fpr(metrics['fpr'])
        
        # True Positive Rate interpretation
        if 'recall' in metrics:
            interpretations['tpr'] = MetricsCalculator._interpret_tpr(metrics['recall'])
        
        return interpretations
    
    @staticmethod
    def _interpret_auc(auc: float) -> str:
        """Interpret AUC score."""
        if auc >= PerformanceConstants.EXCELLENT_AUC:
            return "Excellent discrimination capability"
        elif auc >= PerformanceConstants.GOOD_AUC:
            return "Good discrimination capability"
        elif auc >= PerformanceConstants.FAIR_AUC:
            return "Fair discrimination capability"
        else:
            return "Poor discrimination capability"
    
    @staticmethod
    def _interpret_fpr(fpr: float) -> str:
        """Interpret False Positive Rate."""
        if fpr <= PerformanceConstants.EXCELLENT_FPR:
            return "Excellent false positive control"
        elif fpr <= PerformanceConstants.GOOD_FPR:
            return "Good false positive control"
        else:
            return "High false positive rate - consider adjusting threshold"
    
    @staticmethod
    def _interpret_tpr(tpr: float) -> str:
        """Interpret True Positive Rate."""
        if tpr >= PerformanceConstants.EXCELLENT_TPR:
            return "Excellent anomaly detection capability"
        elif tpr >= PerformanceConstants.GOOD_TPR:
            return "Good anomaly detection capability"
        else:
            return "Moderate anomaly detection capability"


class TrainingMonitor:
    """Utility class for monitoring training progress."""
    
    def __init__(self, patience: int = ModelDefaults.EARLY_STOPPING_PATIENCE):
        """Initialize training monitor.
        
        Args:
            patience: Early stopping patience.
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.training_history = []
    
    def update(self, epoch: int, train_loss: float, val_loss: float = None) -> bool:
        """Update training monitor with new metrics.
        
        Args:
            epoch: Current epoch number.
            train_loss: Training loss.
            val_loss: Validation loss (optional).
            
        Returns:
            True if training should stop (early stopping triggered).
        """
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Use validation loss if available, otherwise training loss
        current_loss = val_loss if val_loss is not None else train_loss
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                return True
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary.
        
        Returns:
            Dictionary with training summary.
        """
        if not self.training_history:
            return {}
        
        final_metrics = self.training_history[-1]
        
        return {
            'total_epochs': len(self.training_history),
            'best_loss': float(self.best_loss),
            'final_train_loss': final_metrics['train_loss'],
            'final_val_loss': final_metrics.get('val_loss'),
            'early_stopped': self.wait >= self.patience,
            'training_history': self.training_history
        }
