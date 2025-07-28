"""FastAPI application for network intrusion detection service."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uvicorn

# Python 3.6 compatibility for asynccontextmanager
try:
    from contextlib import asynccontextmanager
except ImportError:
    # Fallback for Python 3.6
    from contextlib import contextmanager
    import asyncio
    from functools import wraps
    
    def asynccontextmanager(func):
        """Simple asynccontextmanager fallback for Python 3.6."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            generator = func(*args, **kwargs)
            try:
                value = await generator.__anext__()
                yield value
            except StopAsyncIteration:
                pass
            finally:
                try:
                    await generator.__anext__()
                except StopAsyncIteration:
                    pass
        return wrapper

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..utils.logger import get_logger, setup_logging
try:
    from ..utils.config import Config, load_config
except ImportError:
    # Fallback for when pydantic is not available
    try:
        from ..utils.config_manager import SimpleConfigManager as Config
        from ..utils import load_config
    except ImportError:
        # Final fallback
        class Config:
            pass
        def load_config(**kwargs):
            return {}
from ..core.predictor import AnomalyPredictor

# Setup logging
logger = get_logger(__name__)

# Global variables for model and predictor
predictor: Optional[AnomalyPredictor] = None
app_config: Optional[Config] = None

# Pydantic models for API
if FASTAPI_AVAILABLE:
    class NetworkFeatures(BaseModel):
        """Network traffic features for prediction."""
        
        Duration: float = Field(..., description="Connection duration in seconds")
        Orig_bytes: float = Field(..., description="Number of bytes sent by originator")
        Resp_bytes: float = Field(..., description="Number of bytes sent by responder")
        Orig_pkts: float = Field(..., description="Number of packets sent by originator")
        
        @validator('Duration', 'Orig_bytes', 'Resp_bytes', 'Orig_pkts')
        def validate_non_negative(cls, v):
            if v < 0:
                raise ValueError('Feature values must be non-negative')
            return v
        
        class Config:
            schema_extra = {
                "example": {
                    "Duration": 1.5,
                    "Orig_bytes": 1024.0,
                    "Resp_bytes": 512.0,
                    "Orig_pkts": 10.0
                }
            }
    
    
    class BatchPredictionRequest(BaseModel):
        """Batch prediction request."""
        
        samples: List[NetworkFeatures] = Field(..., description="List of network samples")
        threshold_method: str = Field(
            default="percentile", 
            description="Threshold method to use",
            regex="^(percentile|statistical|roc_optimal)$"
        )
        
        class Config:
            schema_extra = {
                "example": {
                    "samples": [
                        {
                            "Duration": 1.5,
                            "Orig_bytes": 1024.0,
                            "Resp_bytes": 512.0,
                            "Orig_pkts": 10.0
                        },
                        {
                            "Duration": 0.1,
                            "Orig_bytes": 100.0,
                            "Resp_bytes": 50.0,
                            "Orig_pkts": 2.0
                        }
                    ],
                    "threshold_method": "percentile"
                }
            }
    
    
    class SinglePredictionRequest(BaseModel):
        """Single prediction request."""
        
        features: NetworkFeatures = Field(..., description="Network traffic features")
        threshold_method: str = Field(
            default="percentile",
            description="Threshold method to use",
            regex="^(percentile|statistical|roc_optimal)$"
        )
        
        class Config:
            schema_extra = {
                "example": {
                    "features": {
                        "Duration": 1.5,
                        "Orig_bytes": 1024.0,
                        "Resp_bytes": 512.0,
                        "Orig_pkts": 10.0
                    },
                    "threshold_method": "percentile"
                }
            }
    
    
    class PredictionResponse(BaseModel):
        """Prediction response."""
        
        is_anomaly: bool = Field(..., description="Whether the sample is anomalous")
        reconstruction_error: float = Field(..., description="Reconstruction error value")
        threshold: float = Field(..., description="Threshold used for detection")
        confidence: float = Field(..., description="Confidence score")
        threshold_method: str = Field(..., description="Threshold method used")
        
        class Config:
            schema_extra = {
                "example": {
                    "is_anomaly": True,
                    "reconstruction_error": 0.85,
                    "threshold": 0.75,
                    "confidence": 1.13,
                    "threshold_method": "percentile"
                }
            }
    
    
    class BatchPredictionResponse(BaseModel):
        """Batch prediction response."""
        
        predictions: List[bool] = Field(..., description="List of anomaly predictions")
        reconstruction_errors: List[float] = Field(..., description="List of reconstruction errors")
        confidences: List[float] = Field(..., description="List of confidence scores")
        threshold: float = Field(..., description="Threshold used for detection")
        threshold_method: str = Field(..., description="Threshold method used")
        statistics: Dict[str, Any] = Field(..., description="Batch statistics")
        inference_time: float = Field(..., description="Total inference time in seconds")
        
        class Config:
            schema_extra = {
                "example": {
                    "predictions": [True, False],
                    "reconstruction_errors": [0.85, 0.45],
                    "confidences": [1.13, 0.60],
                    "threshold": 0.75,
                    "threshold_method": "percentile",
                    "statistics": {
                        "total_samples": 2,
                        "num_anomalies": 1,
                        "anomaly_rate": 0.5,
                        "mean_error": 0.65,
                        "max_error": 0.85,
                        "min_error": 0.45
                    },
                    "inference_time": 0.002
                }
            }
    
    
    class HealthResponse(BaseModel):
        """Health check response."""
        
        status: str = Field(..., description="Service status")
        timestamp: float = Field(..., description="Current timestamp")
        model_loaded: bool = Field(..., description="Whether model is loaded")
        scaler_loaded: bool = Field(..., description="Whether scaler is loaded")
        uptime: float = Field(..., description="Service uptime in seconds")
        
        class Config:
            schema_extra = {
                "example": {
                    "status": "healthy",
                    "timestamp": 1643723400.0,
                    "model_loaded": True,
                    "scaler_loaded": True,
                    "uptime": 3600.0
                }
            }
    
    
    class ModelInfoResponse(BaseModel):
        """Model information response."""
        
        model_info: Dict[str, Any] = Field(..., description="Model information")
        predictor_info: Dict[str, Any] = Field(..., description="Predictor information")
        
        class Config:
            schema_extra = {
                "example": {
                    "model_info": {
                        "model_type": "Autoencoder",
                        "input_size": 4,
                        "hidden_size": 2,
                        "total_parameters": 20,
                        "model_size_mb": 0.00008
                    },
                    "predictor_info": {
                        "model_loaded": True,
                        "scaler_loaded": True,
                        "thresholds_available": ["percentile", "statistical"],
                        "device": "cpu"
                    }
                }
            }


# Global state
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NIDS API service...")
    
    # Load configuration
    global app_config, predictor
    
    config_path = Path("config/api_config.yaml")
    if config_path.exists():
        app_config = load_config(config_path)
    else:
        app_config = Config()
        logger.warning("API config not found. Using defaults.")
    
    # Initialize predictor
    try:
        model_path = Path("models/autoencoder_trained.pth")
        scaler_path = Path("models/scaler.joblib")
        
        if model_path.exists() and scaler_path.exists():
            predictor = AnomalyPredictor(model_path, scaler_path, app_config)
            
            # Optimize thresholds with dummy data if needed
            import numpy as np
            dummy_normal = np.random.randn(100, 4) * 0.1
            predictor.optimize_thresholds(dummy_normal)
            
            logger.info("Predictor initialized successfully")
        else:
            logger.warning("Model or scaler files not found. API will have limited functionality.")
    
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        predictor = None
    
    logger.info("NIDS API service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NIDS API service...")


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Network Intrusion Detection API",
        description="REST API for network anomaly detection using autoencoders",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    def get_predictor() -> AnomalyPredictor:
        """Dependency to get the predictor instance."""
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Predictor not available. Check service health."
            )
        return predictor
    
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Network Intrusion Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        global start_time
        
        return HealthResponse(
            status="healthy" if predictor is not None else "degraded",
            timestamp=time.time(),
            model_loaded=predictor is not None and predictor.model is not None,
            scaler_loaded=predictor is not None and predictor.data_processor is not None,
            uptime=time.time() - start_time
        )
    
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(
        request: SinglePredictionRequest,
        predictor_instance: AnomalyPredictor = Depends(get_predictor)
    ):
        """Predict anomaly for a single network sample."""
        try:
            # Convert features to dictionary
            features_dict = request.features.dict()
            
            # Make prediction
            result = predictor_instance.predict_from_raw_data(
                features_dict, 
                request.threshold_method
            )
            
            return PredictionResponse(**result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
    
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        request: BatchPredictionRequest,
        predictor_instance: AnomalyPredictor = Depends(get_predictor)
    ):
        """Predict anomalies for a batch of network samples."""
        try:
            # Convert samples to numpy array
            import numpy as np
            
            features_list = []
            for sample in request.samples:
                features_dict = sample.dict()
                features_list.append([
                    features_dict["Duration"],
                    features_dict["Orig_bytes"],
                    features_dict["Resp_bytes"],
                    features_dict["Orig_pkts"]
                ])
            
            features_array = np.array(features_list)
            
            # Transform features if data processor is available
            if predictor_instance.data_processor:
                scaled_features = predictor_instance.data_processor.scaler.transform(features_array)
            else:
                scaled_features = features_array
            
            # Make prediction
            result = predictor_instance.predict_batch(
                scaled_features,
                request.threshold_method
            )
            
            return BatchPredictionResponse(**result)
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}"
            )
    
    
    @app.get("/model/info", response_model=ModelInfoResponse)
    async def get_model_info(
        predictor_instance: AnomalyPredictor = Depends(get_predictor)
    ):
        """Get model and predictor information."""
        return ModelInfoResponse(
            model_info=predictor_instance.model_info,
            predictor_info=predictor_instance.get_predictor_info()
        )
    
    
    @app.get("/thresholds")
    async def get_thresholds(
        predictor_instance: AnomalyPredictor = Depends(get_predictor)
    ):
        """Get available detection thresholds."""
        return {
            "thresholds": predictor_instance.thresholds,
            "available_methods": list(predictor_instance.thresholds.keys())
        }
    
    
    @app.post("/thresholds/optimize")
    async def optimize_thresholds(
        background_tasks: BackgroundTasks,
        predictor_instance: AnomalyPredictor = Depends(get_predictor)
    ):
        """Trigger threshold optimization (background task)."""
        
        def optimize_task():
            import numpy as np
            # Generate dummy normal data for optimization
            dummy_normal = np.random.randn(1000, 4) * 0.1
            predictor_instance.optimize_thresholds(dummy_normal)
            logger.info("Threshold optimization completed")
        
        background_tasks.add_task(optimize_task)
        
        return {"message": "Threshold optimization started"}
    
    
    @app.get("/metrics")
    async def get_metrics():
        """Get basic metrics (Prometheus-compatible format)."""
        metrics = []
        
        if predictor is not None:
            metrics.append("# HELP nids_model_loaded Model loading status")
            metrics.append("# TYPE nids_model_loaded gauge")
            metrics.append(f"nids_model_loaded {1 if predictor.model else 0}")
            
            metrics.append("# HELP nids_scaler_loaded Scaler loading status")
            metrics.append("# TYPE nids_scaler_loaded gauge")
            metrics.append(f"nids_scaler_loaded {1 if predictor.data_processor else 0}")
        
        metrics.append("# HELP nids_uptime_seconds Service uptime")
        metrics.append("# TYPE nids_uptime_seconds counter")
        metrics.append(f"nids_uptime_seconds {time.time() - start_time}")
        
        return Response(content="\n".join(metrics), media_type="text/plain")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
) -> None:
    """Start the API server.
    
    Args:
        host: Host to bind to.
        port: Port to bind to.
        workers: Number of worker processes.
        reload: Enable auto-reload for development.
        log_level: Logging level.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    # Setup logging
    setup_logging(level=log_level.upper())
    
    logger.info(f"Starting NIDS API server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


def main():
    """Main entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Intrusion Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
