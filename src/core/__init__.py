"""Core modules for the NIDS system."""

# Import order matters for proper resolution
try:
    from .autoencoder import AutoencoderModel, SimpleNumpyAutoencoder
    from .trainer import ModelTrainer
    from .predictor import AnomalyPredictor, ThresholdOptimizer
    
    __all__ = [
        "AutoencoderModel",
        "SimpleNumpyAutoencoder", 
        "ModelTrainer",
        "AnomalyPredictor",
        "ThresholdOptimizer",
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some core modules could not be imported: {e}")
    
    __all__ = []
