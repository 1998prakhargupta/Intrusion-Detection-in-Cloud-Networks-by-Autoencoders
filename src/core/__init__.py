"""Enhanced core modules for the NIDS system."""

# Import order matters for proper resolution
try:
    from ..models.autoencoder import AutoencoderModel, SimpleNumpyAutoencoder
    from .trainer import ModelTrainer
    from .predictor import AnomalyPredictor, ThresholdOptimizer
    from .enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
    from .evaluator import ModelEvaluator, ThresholdCalculator, MetricsCalculator
    
    __all__ = [
        "AutoencoderModel",
        "SimpleNumpyAutoencoder", 
        "ModelTrainer",
        "AnomalyPredictor",
        "ThresholdOptimizer",
        "EnhancedModelTrainer",
        "ProductionAutoencoder",
        "ModelEvaluator",
        "ThresholdCalculator",
        "MetricsCalculator",
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some core modules could not be imported: {e}")
    
    __all__ = []
