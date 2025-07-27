"""Prediction and anomaly detection functionality."""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import LoggerMixin
from ..utils.config import Config
from ..data.processor import DataProcessor
from .autoencoder import AutoencoderModel, SimpleNumpyAutoencoder


class ThresholdOptimizer:
    """Utility class for optimizing anomaly detection thresholds."""
    
    @staticmethod
    def percentile_threshold(normal_errors: np.ndarray, percentile: float = 95) -> float:
        """Calculate threshold based on percentile of normal errors.
        
        Args:
            normal_errors: Reconstruction errors from normal data.
            percentile: Percentile value (0-100).
            
        Returns:
            Threshold value.
        """
        return np.percentile(normal_errors, percentile)
    
    @staticmethod
    def statistical_threshold(normal_errors: np.ndarray, n_std: float = 2.0) -> float:
        """Calculate threshold based on statistical method (mean + n*std).
        
        Args:
            normal_errors: Reconstruction errors from normal data.
            n_std: Number of standard deviations.
            
        Returns:
            Threshold value.
        """
        return np.mean(normal_errors) + n_std * np.std(normal_errors)
    
    @staticmethod
    def roc_optimal_threshold(normal_errors: np.ndarray, 
                            anomalous_errors: np.ndarray,
                            optimization_metric: str = "f1") -> float:
        """Calculate optimal threshold using ROC analysis.
        
        Args:
            normal_errors: Reconstruction errors from normal data.
            anomalous_errors: Reconstruction errors from anomalous data.
            optimization_metric: Metric to optimize ('f1', 'youden', 'precision', 'recall').
            
        Returns:
            Optimal threshold value.
        """
        # Combine errors and create labels
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        all_labels = np.concatenate([
            np.zeros(len(normal_errors)), 
            np.ones(len(anomalous_errors))
        ])
        
        # Sort by error values
        sorted_indices = np.argsort(all_errors)
        sorted_errors = all_errors[sorted_indices]
        sorted_labels = all_labels[sorted_indices]
        
        best_threshold = 0
        best_score = -1
        
        # Sample thresholds for efficiency
        sample_indices = np.linspace(0, len(sorted_errors) - 1, min(100, len(sorted_errors)), dtype=int)
        
        for i in sample_indices:
            threshold = sorted_errors[i]
            
            # Calculate metrics
            predicted = (all_errors > threshold).astype(int)
            tp = np.sum((all_labels == 1) & (predicted == 1))
            fn = np.sum((all_labels == 1) & (predicted == 0))
            fp = np.sum((all_labels == 0) & (predicted == 1))
            tn = np.sum((all_labels == 0) & (predicted == 0))
            
            # Calculate performance metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate optimization score
            if optimization_metric == "f1":
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            elif optimization_metric == "youden":
                score = recall + specificity - 1
            elif optimization_metric == "precision":
                score = precision
            elif optimization_metric == "recall":
                score = recall
            else:
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold


class AnomalyPredictor(LoggerMixin):
    """Anomaly predictor for network intrusion detection."""
    
    def __init__(self, 
                 model_path: Optional[Union[str, Path]] = None,
                 scaler_path: Optional[Union[str, Path]] = None,
                 config: Optional[Config] = None):
        """Initialize anomaly predictor.
        
        Args:
            model_path: Path to trained model.
            scaler_path: Path to fitted scaler.
            config: Configuration object.
        """
        super().__init__()
        self.config = config or Config()
        self.model = None
        self.data_processor = None
        self.thresholds = {}
        self.device = None
        self.model_info = {}
        
        # Setup device
        self._setup_device()
        
        # Load model and scaler if provided
        if model_path:
            self.load_model(model_path)
        
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def _setup_device(self) -> None:
        """Setup compute device."""
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
        else:
            self.device = "cpu"
            self.logger.info("PyTorch not available. Using NumPy implementation.")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load trained model.
        
        Args:
            model_path: Path to the model file.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if TORCH_AVAILABLE:
            try:
                self.model = AutoencoderModel.load_model(model_path, self.device)
                self.model_info = self.model.get_model_info()
                self.logger.info(f"PyTorch model loaded from {model_path}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load as PyTorch model: {e}")
        
        # Try to load as NumPy model
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_info = {
                "model_type": "NumPy Autoencoder",
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size
            }
            
            self.logger.info(f"NumPy model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """Load data scaler.
        
        Args:
            scaler_path: Path to the scaler file.
        """
        self.data_processor = DataProcessor(self.config)
        self.data_processor.load_scaler(scaler_path)
        self.logger.info(f"Scaler loaded from {scaler_path}")
    
    def calculate_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for input data.
        
        Args:
            data: Input data array.
            
        Returns:
            Array of reconstruction errors.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load model first.")
        
        if TORCH_AVAILABLE and isinstance(self.model, AutoencoderModel):
            # PyTorch model
            self.model.eval()
            with torch.no_grad():
                data_tensor = torch.FloatTensor(data).to(self.device)
                errors = self.model.reconstruction_error(data_tensor)
                return errors.cpu().numpy()
        else:
            # NumPy model
            return self.model.reconstruction_error(data)
    
    def optimize_thresholds(self, 
                          normal_data: np.ndarray, 
                          anomalous_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Optimize detection thresholds using normal and optionally anomalous data.
        
        Args:
            normal_data: Normal data for threshold calculation.
            anomalous_data: Anomalous data (optional, for ROC-based optimization).
            
        Returns:
            Dictionary of optimized thresholds.
        """
        self.logger.info("Optimizing detection thresholds...")
        
        # Calculate reconstruction errors for normal data
        normal_errors = self.calculate_reconstruction_errors(normal_data)
        
        thresholds = {}
        
        # Percentile-based threshold
        if "percentile" in self.config.thresholds.methods:
            thresholds["percentile"] = ThresholdOptimizer.percentile_threshold(
                normal_errors, self.config.thresholds.percentile_value
            )
        
        # Statistical threshold
        if "statistical" in self.config.thresholds.methods:
            thresholds["statistical"] = ThresholdOptimizer.statistical_threshold(
                normal_errors, self.config.thresholds.statistical_n_std
            )
        
        # ROC-optimal threshold (requires anomalous data)
        if "roc_optimal" in self.config.thresholds.methods and anomalous_data is not None:
            anomalous_errors = self.calculate_reconstruction_errors(anomalous_data)
            thresholds["roc_optimal"] = ThresholdOptimizer.roc_optimal_threshold(
                normal_errors, anomalous_errors, 
                self.config.thresholds.roc_optimization_metric
            )
        
        self.thresholds = thresholds
        
        self.logger.info(f"Thresholds optimized: {thresholds}")
        return thresholds
    
    def predict_single(self, 
                      features: Union[np.ndarray, List[float]], 
                      threshold_method: str = "percentile") -> Dict[str, Any]:
        """Predict anomaly for a single sample.
        
        Args:
            features: Input features.
            threshold_method: Threshold method to use.
            
        Returns:
            Prediction result dictionary.
        """
        if isinstance(features, list):
            features = np.array(features)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Calculate reconstruction error
        error = self.calculate_reconstruction_errors(features)[0]
        
        # Get threshold
        if threshold_method not in self.thresholds:
            raise ValueError(f"Threshold method '{threshold_method}' not available. "
                           f"Available methods: {list(self.thresholds.keys())}")
        
        threshold = self.thresholds[threshold_method]
        
        # Make prediction
        is_anomaly = error > threshold
        confidence = min(error / threshold, 5.0) if threshold > 0 else 1.0  # Cap at 5x threshold
        
        return {
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": float(error),
            "threshold": float(threshold),
            "confidence": float(confidence),
            "threshold_method": threshold_method
        }
    
    def predict_batch(self, 
                     data: np.ndarray, 
                     threshold_method: str = "percentile") -> Dict[str, Any]:
        """Predict anomalies for a batch of samples.
        
        Args:
            data: Input data array.
            threshold_method: Threshold method to use.
            
        Returns:
            Batch prediction results.
        """
        start_time = time.time()
        
        # Calculate reconstruction errors
        errors = self.calculate_reconstruction_errors(data)
        
        # Get threshold
        if threshold_method not in self.thresholds:
            raise ValueError(f"Threshold method '{threshold_method}' not available. "
                           f"Available methods: {list(self.thresholds.keys())}")
        
        threshold = self.thresholds[threshold_method]
        
        # Make predictions
        predictions = errors > threshold
        confidences = np.minimum(errors / threshold if threshold > 0 else np.ones_like(errors), 5.0)
        
        # Calculate statistics
        num_anomalies = np.sum(predictions)
        anomaly_rate = num_anomalies / len(predictions)
        
        inference_time = time.time() - start_time
        
        return {
            "predictions": predictions.tolist(),
            "reconstruction_errors": errors.tolist(),
            "confidences": confidences.tolist(),
            "threshold": float(threshold),
            "threshold_method": threshold_method,
            "statistics": {
                "total_samples": len(predictions),
                "num_anomalies": int(num_anomalies),
                "anomaly_rate": float(anomaly_rate),
                "mean_error": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
                "min_error": float(np.min(errors))
            },
            "inference_time": inference_time
        }
    
    def predict_from_raw_data(self, 
                            raw_data: Union[np.ndarray, Dict[str, Any]], 
                            threshold_method: str = "percentile") -> Dict[str, Any]:
        """Predict anomalies from raw network traffic data.
        
        Args:
            raw_data: Raw network data (requires data processor).
            threshold_method: Threshold method to use.
            
        Returns:
            Prediction results.
        """
        if self.data_processor is None:
            raise ValueError("Data processor not loaded. Load scaler first.")
        
        # Process raw data
        if isinstance(raw_data, dict):
            # Convert dict to numpy array using feature names
            features = []
            for feature_name in self.data_processor.feature_names:
                if feature_name in raw_data:
                    features.append(raw_data[feature_name])
                else:
                    self.logger.warning(f"Feature '{feature_name}' not found in input data")
                    features.append(0.0)  # Default value
            
            features = np.array(features).reshape(1, -1)
        else:
            features = raw_data
        
        # Transform features using the loaded scaler
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Apply feature scaling
        scaled_features = self.data_processor.scaler.transform(features)
        
        # Make prediction
        if scaled_features.shape[0] == 1:
            return self.predict_single(scaled_features[0], threshold_method)
        else:
            return self.predict_batch(scaled_features, threshold_method)
    
    def evaluate_performance(self, 
                           normal_data: np.ndarray, 
                           anomalous_data: np.ndarray,
                           threshold_method: str = "percentile") -> Dict[str, Any]:
        """Evaluate model performance on labeled data.
        
        Args:
            normal_data: Normal data samples.
            anomalous_data: Anomalous data samples.
            threshold_method: Threshold method to use.
            
        Returns:
            Performance metrics.
        """
        self.logger.info("Evaluating model performance...")
        
        # Calculate reconstruction errors
        normal_errors = self.calculate_reconstruction_errors(normal_data)
        anomalous_errors = self.calculate_reconstruction_errors(anomalous_data)
        
        # Combine data and labels
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        true_labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomalous_errors))])
        
        # Get threshold and make predictions
        threshold = self.thresholds.get(threshold_method)
        if threshold is None:
            # Use percentile threshold as fallback
            threshold = np.percentile(normal_errors, 95)
        
        predictions = (all_errors > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((true_labels == 1) & (predictions == 1))
        tn = np.sum((true_labels == 0) & (predictions == 0))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate ROC AUC (simplified)
        roc_auc = self._calculate_auc(normal_errors, anomalous_errors)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            },
            "error_statistics": {
                "normal_mean": float(np.mean(normal_errors)),
                "normal_std": float(np.std(normal_errors)),
                "anomalous_mean": float(np.mean(anomalous_errors)),
                "anomalous_std": float(np.std(anomalous_errors))
            }
        }
    
    def _calculate_auc(self, normal_errors: np.ndarray, anomalous_errors: np.ndarray) -> float:
        """Calculate AUC score manually.
        
        Args:
            normal_errors: Normal reconstruction errors.
            anomalous_errors: Anomalous reconstruction errors.
            
        Returns:
            AUC score.
        """
        all_errors = np.concatenate([normal_errors, anomalous_errors])
        all_labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomalous_errors))])
        
        # Sort by error values
        sorted_indices = np.argsort(all_errors)
        sorted_labels = all_labels[sorted_indices]
        
        tpr_values = []
        fpr_values = []
        
        # Sample points for efficiency
        sample_indices = np.linspace(0, len(sorted_labels) - 1, min(100, len(sorted_labels)), dtype=int)
        
        for i in sample_indices:
            threshold_idx = i
            predicted = np.zeros(len(sorted_labels))
            predicted[threshold_idx:] = 1
            
            tp = np.sum((sorted_labels == 1) & (predicted == 1))
            fn = np.sum((sorted_labels == 1) & (predicted == 0))
            fp = np.sum((sorted_labels == 0) & (predicted == 1))
            tn = np.sum((sorted_labels == 0) & (predicted == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return abs(auc)
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """Get predictor information and status.
        
        Returns:
            Predictor information dictionary.
        """
        return {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.data_processor is not None,
            "thresholds_available": list(self.thresholds.keys()),
            "model_info": self.model_info,
            "device": str(self.device),
            "feature_names": self.data_processor.feature_names if self.data_processor else None
        }
