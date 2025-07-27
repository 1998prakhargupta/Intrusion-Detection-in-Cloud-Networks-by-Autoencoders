#!/bin/bash

# Model training and evaluation script
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DATA_PATH="${DATA_PATH:-dataset/CIDDS-001-external-week3_1.csv}"
CONFIG_PATH="${CONFIG_PATH:-config/training.yaml}"
MODEL_OUTPUT_PATH="${MODEL_OUTPUT_PATH:-models/}"
PYTHON_CMD="${PYTHON_CMD:-python}"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if data file exists
check_data() {
    log "Checking data availability..."
    
    if [ ! -f "$DATA_PATH" ]; then
        error "Data file not found: $DATA_PATH"
        exit 1
    fi
    
    # Check file size
    FILE_SIZE=$(stat -c%s "$DATA_PATH" 2>/dev/null || stat -f%z "$DATA_PATH" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1024 ]; then
        error "Data file appears to be empty or too small"
        exit 1
    fi
    
    success "Data file found: $DATA_PATH ($(du -h "$DATA_PATH" | cut -f1))"
}

# Train the model
train_model() {
    log "Starting model training..."
    
    cat << 'EOF' > train_script.py
import sys
import os
sys.path.append('src')

from src.utils.config import load_config
from src.data.processor import DataProcessor
from src.core.trainer import ModelTrainer
from src.utils.logger import setup_logging
import logging

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config('config/training.yaml')
        logger.info("Configuration loaded successfully")
        
        # Initialize data processor
        processor = DataProcessor(config.data)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = processor.load_data('dataset/CIDDS-001-external-week3_1.csv')
        X_train, X_val = processor.preprocess(data)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Initialize trainer
        trainer = ModelTrainer(config.model)
        
        # Train model
        logger.info("Starting model training...")
        metrics = trainer.train(X_train, X_val)
        
        # Save model
        model_path = f"models/autoencoder_{metrics['final_epoch']}.pth"
        trainer.save_model(model_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Final validation loss: {metrics['final_val_loss']:.4f}")
        logger.info(f"Training completed in {metrics['training_time']:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    $PYTHON_CMD train_script.py
    TRAIN_STATUS=$?
    
    # Cleanup
    rm -f train_script.py
    
    if [ $TRAIN_STATUS -eq 0 ]; then
        success "Model training completed successfully"
    else
        error "Model training failed"
        exit 1
    fi
}

# Evaluate the model
evaluate_model() {
    log "Evaluating model performance..."
    
    cat << 'EOF' > eval_script.py
import sys
import os
sys.path.append('src')

from src.utils.config import load_config
from src.data.processor import DataProcessor
from src.core.predictor import AnomalyPredictor
from src.utils.logger import setup_logging
import logging
import glob

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config('config/training.yaml')
        
        # Find the latest model
        model_files = glob.glob('models/autoencoder_*.pth')
        if not model_files:
            logger.error("No trained models found")
            return False
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"Using model: {latest_model}")
        
        # Initialize components
        processor = DataProcessor(config.data)
        predictor = AnomalyPredictor(config.model)
        
        # Load test data
        data = processor.load_data('dataset/CIDDS-001-external-week3_1.csv')
        X_test, _ = processor.preprocess(data)
        
        # Load model and predict
        predictor.load_model(latest_model)
        predictions = predictor.predict(X_test[:1000])  # Test on subset
        
        # Calculate metrics
        anomaly_rate = (predictions == 1).mean()
        
        logger.info(f"Test samples: {len(predictions)}")
        logger.info(f"Anomaly rate: {anomaly_rate:.3f}")
        logger.info(f"Normal rate: {1-anomaly_rate:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    $PYTHON_CMD eval_script.py
    EVAL_STATUS=$?
    
    # Cleanup
    rm -f eval_script.py
    
    if [ $EVAL_STATUS -eq 0 ]; then
        success "Model evaluation completed"
    else
        error "Model evaluation failed"
        exit 1
    fi
}

# Create model package
package_model() {
    log "Creating model package..."
    
    # Create deployment package
    PACKAGE_NAME="nids-model-$(date +%Y%m%d-%H%M%S)"
    PACKAGE_DIR="packages/$PACKAGE_NAME"
    
    mkdir -p "$PACKAGE_DIR"
    
    # Copy model files
    cp -r models/*.pth "$PACKAGE_DIR/" 2>/dev/null || true
    cp -r config/ "$PACKAGE_DIR/"
    
    # Create deployment info
    cat > "$PACKAGE_DIR/deployment-info.json" << EOF
{
    "package_name": "$PACKAGE_NAME",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "data_source": "$DATA_PATH",
    "config_used": "$CONFIG_PATH",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "python_version": "$($PYTHON_CMD --version 2>&1)"
}
EOF
    
    # Create package tarball
    tar -czf "${PACKAGE_NAME}.tar.gz" -C packages "$PACKAGE_NAME"
    
    success "Model package created: ${PACKAGE_NAME}.tar.gz"
}

# Main function
main() {
    log "Starting NIDS model training pipeline..."
    
    check_data
    train_model
    evaluate_model
    package_model
    
    success "Training pipeline completed successfully!"
    
    echo ""
    log "Generated artifacts:"
    echo "  Models: $(ls models/*.pth 2>/dev/null | wc -l) files"
    echo "  Package: $(ls *.tar.gz 2>/dev/null | tail -1)"
    echo ""
    log "Next steps:"
    echo "  1. Review model performance in logs"
    echo "  2. Test the model with: python -m src.api.main"
    echo "  3. Deploy with: ./scripts/deploy.sh"
}

# Handle script arguments
case "${1:-train}" in
    "train")
        main
        ;;
    "evaluate")
        check_data
        evaluate_model
        ;;
    "package")
        package_model
        ;;
    *)
        echo "Usage: $0 {train|evaluate|package}"
        echo "  train    - Full training pipeline (default)"
        echo "  evaluate - Evaluate existing model"
        echo "  package  - Package model for deployment"
        exit 1
        ;;
esac
