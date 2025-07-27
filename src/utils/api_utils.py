"""API utilities for request handling and response formatting."""

import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

from .constants import APIConstants, ErrorMessages
from .logger import get_logger

logger = get_logger(__name__)


class ResponseFormatter:
    """Utility class for formatting API responses."""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success", 
                        request_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a success response.
        
        Args:
            data: Response data.
            message: Success message.
            request_id: Request ID for tracking.
            
        Returns:
            Formatted success response.
        """
        response = {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if request_id:
            response["request_id"] = request_id
        
        return response
    
    @staticmethod
    def error_response(error_code: str, message: str, 
                      details: Optional[str] = None,
                      request_id: Optional[str] = None) -> Dict[str, Any]:
        """Create an error response.
        
        Args:
            error_code: Error code identifier.
            message: Error message.
            details: Additional error details.
            request_id: Request ID for tracking.
            
        Returns:
            Formatted error response.
        """
        response = {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if details:
            response["details"] = details
        
        if request_id:
            response["request_id"] = request_id
        
        return response
    
    @staticmethod
    def prediction_response(predictions: List[Dict[str, Any]], 
                          model_info: Dict[str, Any],
                          processing_time: float,
                          request_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a prediction response.
        
        Args:
            predictions: List of prediction results.
            model_info: Information about the model used.
            processing_time: Time taken to process the request.
            request_id: Request ID for tracking.
            
        Returns:
            Formatted prediction response.
        """
        return ResponseFormatter.success_response(
            data={
                "predictions": predictions,
                "model_info": model_info,
                "processing_time_seconds": round(processing_time, 4),
                "prediction_count": len(predictions)
            },
            message="Predictions generated successfully",
            request_id=request_id
        )


class RequestValidator:
    """Utility class for validating API requests."""
    
    @staticmethod
    def validate_prediction_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a prediction request.
        
        Args:
            data: Request data to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check required fields
        required_fields = ["network_data"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate network_data
        network_data = data["network_data"]
        if not isinstance(network_data, (list, dict)):
            return False, "network_data must be a list or dictionary"
        
        # If list, check each item
        if isinstance(network_data, list):
            if len(network_data) == 0:
                return False, "network_data cannot be empty"
            
            if len(network_data) > APIConstants.MAX_BATCH_SIZE:
                return False, f"Batch size exceeds maximum allowed ({APIConstants.MAX_BATCH_SIZE})"
            
            for i, item in enumerate(network_data):
                if not isinstance(item, dict):
                    return False, f"network_data[{i}] must be a dictionary"
        
        return True, None
    
    @staticmethod
    def validate_training_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a training request.
        
        Args:
            data: Request data to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check required fields
        required_fields = ["data_path"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate data_path
        data_path = Path(data["data_path"])
        if not data_path.exists():
            return False, f"Data file not found: {data_path}"
        
        # Validate optional parameters
        if "epochs" in data:
            epochs = data["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                return False, "epochs must be a positive integer"
        
        if "batch_size" in data:
            batch_size = data["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                return False, "batch_size must be a positive integer"
        
        return True, None
    
    @staticmethod
    def validate_model_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a model management request.
        
        Args:
            data: Request data to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        if "model_name" in data:
            model_name = data["model_name"]
            if not isinstance(model_name, str) or len(model_name.strip()) == 0:
                return False, "model_name must be a non-empty string"
        
        return True, None


class DataConverter:
    """Utility class for converting data between formats."""
    
    @staticmethod
    def numpy_to_json_serializable(obj: Any) -> Any:
        """Convert numpy objects to JSON serializable format.
        
        Args:
            obj: Object to convert.
            
        Returns:
            JSON serializable object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: DataConverter.numpy_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataConverter.numpy_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def format_prediction_result(prediction: Dict[str, Any], 
                               include_details: bool = True) -> Dict[str, Any]:
        """Format a single prediction result.
        
        Args:
            prediction: Raw prediction result.
            include_details: Whether to include detailed information.
            
        Returns:
            Formatted prediction result.
        """
        result = {
            "is_anomaly": bool(prediction.get("is_anomaly", False)),
            "confidence": float(prediction.get("confidence", 0.0)),
            "anomaly_score": float(prediction.get("anomaly_score", 0.0))
        }
        
        if include_details:
            if "reconstruction_error" in prediction:
                result["reconstruction_error"] = float(prediction["reconstruction_error"])
            
            if "threshold" in prediction:
                result["threshold"] = float(prediction["threshold"])
            
            if "network_category" in prediction:
                result["network_category"] = str(prediction["network_category"])
            
            if "timestamp" in prediction:
                result["timestamp"] = str(prediction["timestamp"])
        
        return DataConverter.numpy_to_json_serializable(result)
    
    @staticmethod
    def format_model_info(model_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format model information.
        
        Args:
            model_path: Path to the model file.
            metadata: Model metadata.
            
        Returns:
            Formatted model information.
        """
        info = {
            "model_name": model_path.stem,
            "model_file": model_path.name,
            "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2)
        }
        
        # Add metadata if available
        if metadata:
            if "saved_at" in metadata:
                info["created_at"] = metadata["saved_at"]
            
            if "config" in metadata:
                config = metadata["config"]
                info["architecture"] = {
                    "input_dim": config.get("input_dim"),
                    "hidden_dims": config.get("hidden_dims"),
                    "learning_rate": config.get("learning_rate")
                }
            
            if "metrics" in metadata:
                info["performance"] = DataConverter.numpy_to_json_serializable(
                    metadata["metrics"]
                )
        
        return DataConverter.numpy_to_json_serializable(info)


class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = APIConstants.RATE_LIMIT_REQUESTS,
                 time_window: int = APIConstants.RATE_LIMIT_WINDOW):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed.
            time_window: Time window in seconds.
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier.
            
        Returns:
            True if request is allowed, False otherwise.
        """
        current_time = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.time_window
            ]
        else:
            self.requests[client_id] = []
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True
    
    def get_reset_time(self, client_id: str) -> Optional[float]:
        """Get time when rate limit resets for client.
        
        Args:
            client_id: Client identifier.
            
        Returns:
            Reset time or None if no requests recorded.
        """
        if client_id not in self.requests or not self.requests[client_id]:
            return None
        
        oldest_request = min(self.requests[client_id])
        return oldest_request + self.time_window


class HealthChecker:
    """Utility class for health checks."""
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """Perform system health check.
        
        Returns:
            Health check results.
        """
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "checks": {}
        }
        
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_status["checks"]["memory"] = {
                "status": "ok" if memory.percent < 90 else "warning",
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2)
            }
        except ImportError:
            health_status["checks"]["memory"] = {
                "status": "unknown",
                "message": "psutil not available"
            }
        
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage("/")
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            health_status["checks"]["disk"] = {
                "status": "ok" if disk_percent < 90 else "warning",
                "usage_percent": round(disk_percent, 2),
                "free_gb": round(disk_usage.free / (1024**3), 2)
            }
        except Exception as e:
            health_status["checks"]["disk"] = {
                "status": "error",
                "message": str(e)
            }
        
        # Overall status
        check_statuses = [check["status"] for check in health_status["checks"].values()]
        if "error" in check_statuses:
            health_status["status"] = "unhealthy"
        elif "warning" in check_statuses:
            health_status["status"] = "degraded"
        
        return health_status
    
    @staticmethod
    def check_model_availability(model_paths: List[Path]) -> Dict[str, Any]:
        """Check availability of models.
        
        Args:
            model_paths: List of model file paths to check.
            
        Returns:
            Model availability status.
        """
        model_status = {
            "total_models": len(model_paths),
            "available_models": 0,
            "models": {}
        }
        
        for model_path in model_paths:
            model_name = model_path.stem
            if model_path.exists():
                model_status["models"][model_name] = {
                    "status": "available",
                    "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                    "last_modified": time.strftime(
                        '%Y-%m-%d %H:%M:%S',
                        time.localtime(model_path.stat().st_mtime)
                    )
                }
                model_status["available_models"] += 1
            else:
                model_status["models"][model_name] = {
                    "status": "missing",
                    "path": str(model_path)
                }
        
        return model_status
