#!/usr/bin/env python3
"""
NIDS Autoencoder - Basic Usage Example

This script demonstrates basic usage of the NIDS autoencoder system
for network intrusion detection.

Requirements:
    - Python 3.8+
    - Installed nids-autoencoder package
    - Sample network data

Usage:
    python examples/basic_usage.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.training import AutoencoderTrainer
from src.core.prediction import AutoencoderPredictor
from src.data.preprocessing import DataPreprocessor
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)

def load_sample_data():
    """Load sample network traffic data for demonstration."""
    logger.info("Loading sample network data...")
    
    # Create synthetic data for demonstration
    # In production, load actual network traffic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Generate normal traffic (majority)
    normal_data = np.random.normal(0, 1, (int(n_samples * 0.9), n_features))
    
    # Generate anomalous traffic (minority)
    anomaly_data = np.random.normal(3, 2, (int(n_samples * 0.1), n_features))
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])
    
    # Create DataFrame with feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    logger.info(f"Loaded {len(df)} samples with {n_features} features")
    logger.info(f"Normal samples: {sum(y == 0)}, Anomalous samples: {sum(y == 1)}")
    
    return df

def preprocess_data(df):
    """Preprocess the network data for training."""
    logger.info("Preprocessing network data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    logger.info(f"Data preprocessed: {X_processed.shape}")
    
    return X_processed, y, preprocessor

def train_autoencoder(X_train, config):
    """Train the autoencoder model."""
    logger.info("Training autoencoder model...")
    
    # Initialize trainer
    trainer = AutoencoderTrainer(config)
    
    # Train the model
    model, training_history = trainer.train(X_train)
    
    logger.info("Training completed successfully")
    logger.info(f"Final training loss: {training_history['loss'][-1]:.4f}")
    
    return model, training_history

def detect_anomalies(model, X_test, y_test, preprocessor):
    """Detect anomalies using the trained model."""
    logger.info("Detecting anomalies...")
    
    # Initialize predictor
    predictor = AutoencoderPredictor(model, preprocessor)
    
    # Get anomaly scores
    anomaly_scores = predictor.predict_anomaly_scores(X_test)
    
    # Get binary predictions (normal/anomaly)
    predictions = predictor.predict(X_test)
    
    # Calculate performance metrics
    from sklearn.metrics import classification_report, roc_auc_score
    
    auc_score = roc_auc_score(y_test, anomaly_scores)
    
    logger.info(f"ROC-AUC Score: {auc_score:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, 
                                    target_names=['Normal', 'Anomaly']))
    
    return anomaly_scores, predictions

def main():
    """Main function demonstrating basic usage."""
    logger.info("🚀 Starting NIDS Autoencoder Basic Usage Example")
    
    try:
        # 1. Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 2. Load and preprocess data
        df = load_sample_data()
        X_processed, y, preprocessor = preprocess_data(df)
        
        # 3. Split data for training and testing
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # For autoencoder training, use only normal samples
        X_train_normal = X_train[y_train == 0]
        
        logger.info(f"Training samples (normal only): {len(X_train_normal)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # 4. Train autoencoder
        model, history = train_autoencoder(X_train_normal, config)
        
        # 5. Detect anomalies
        anomaly_scores, predictions = detect_anomalies(
            model, X_test, y_test, preprocessor
        )
        
        # 6. Display results
        logger.info("\n" + "="*50)
        logger.info("RESULTS SUMMARY")
        logger.info("="*50)
        
        n_normal = sum(predictions == 0)
        n_anomaly = sum(predictions == 1)
        
        logger.info(f"Total samples analyzed: {len(predictions)}")
        logger.info(f"Predicted normal: {n_normal}")
        logger.info(f"Predicted anomalies: {n_anomaly}")
        logger.info(f"Anomaly rate: {n_anomaly/len(predictions)*100:.2f}%")
        
        # Show top anomalies
        top_anomalies_idx = np.argsort(anomaly_scores)[-5:]
        logger.info("\nTop 5 Anomaly Scores:")
        for i, idx in enumerate(top_anomalies_idx[::-1], 1):
            score = anomaly_scores[idx]
            actual = "Anomaly" if y_test.iloc[idx] == 1 else "Normal"
            logger.info(f"  {i}. Score: {score:.4f} (Actual: {actual})")
        
        logger.info("\n✅ Basic usage example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in basic usage example: {str(e)}")
        raise

if __name__ == "__main__":
    main()
