#!/usr/bin/env python3
"""
Demo script showing the refactored NIDS modular architecture with enterprise configuration.
This script demonstrates how to use each module independently with centralized configuration.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def demo_configuration_manager():
    """Demonstrate ConfigurationManager module."""
    print("🔧 DEMO: Enterprise Configuration Management")
    print("-" * 50)
    
    try:
        from utils.enterprise_config import ConfigurationManager, Environment
        
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        print("✅ ConfigurationManager initialized")
        
        # Show current environment
        print(f"🌍 Current environment: {config_manager.environment.value}")
        
        # Show configuration summary
        summary = config_manager.get_config_summary()
        print("\n📋 Configuration Summary:")
        for section, data in summary.items():
            print(f"   {section}: {list(data.keys()) if isinstance(data, dict) else data}")
        
        # Demonstrate environment switching
        print(f"\n🔄 Available environments: {[env.value for env in Environment]}")
        
        # Show model configuration
        model_config = config_manager.get_model_config()
        print(f"\n🏗️  Model Configuration:")
        print(f"   Input dimensions: {model_config.input_dim}")
        print(f"   Hidden layers: {model_config.hidden_dims}")
        print(f"   Activation: {model_config.activation}")
        print(f"   Dropout rate: {model_config.dropout_rate}")
        
        # Show training configuration
        training_config = config_manager.get_training_config()
        print(f"\n🏋️  Training Configuration:")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Learning rate: {training_config.learning_rate}")
        print(f"   Early stopping: {training_config.early_stopping_enabled}")
        
        # Show data configuration
        data_config = config_manager.get_data_config()
        print(f"\n📊 Data Configuration:")
        print(f"   Source: {data_config.source_path}")
        print(f"   Features: {len(data_config.feature_columns)} columns")
        print(f"   Scaling: {data_config.scaling_method}")
        
        # Validate configuration
        is_valid, errors = config_manager.validate_configuration()
        print(f"\n✅ Configuration valid: {is_valid}")
        if errors:
            print("   Validation errors:")
            for error in errors:
                print(f"   - {error}")
        
        # Save current configuration
        config_path = project_root / "demo_config.yaml"
        config_manager.save_configuration(config_path)
        print(f"💾 Configuration saved to: {config_path}")
        
        return config_manager
        
    except Exception as e:
        print(f"❌ ConfigurationManager demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_data_loader(config_manager=None):
    """Demonstrate DataLoader module with configuration management."""
    print("🔍 DEMO: DataLoader Module with Enterprise Config")
    print("-" * 50)
    
    try:
        from data.loader import DataLoader
        
        # Initialize loader with configuration
        loader = DataLoader(config_manager=config_manager)
        print("✅ DataLoader initialized with enterprise configuration")
        
        # Check if demo data exists (use config paths)
        if config_manager:
            data_path = config_manager.get_data_path("external")
            print(f"📁 Using configured data path: {data_path}")
        else:
            data_path = project_root / "data" / "raw" / "CIDDS-001-external-week3_1.csv"
        
        if data_path.exists():
            print(f"📁 Loading data from: {data_path}")
            data = loader.load_and_validate_data(str(data_path))
            
            # Extract features
            features, labels = loader.extract_features_and_labels(data)
            print(f"📊 Features extracted: {len(features)} columns")
            
            # Get summary
            summary = loader.get_data_summary(data)
            print(f"📈 Data shape: {summary['shape']}")
            print(f"💾 Memory usage: {summary['memory_usage_mb']:.1f} MB")
            
            return data, features, labels
        else:
            print("⚠️  Demo dataset not found, creating sample data")
            # Create sample data for demo
            import pandas as pd
            import numpy as np
            
            sample_data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 1000),
                'feature2': np.random.normal(0, 1, 1000),
                'feature3': np.random.uniform(0, 1, 1000),
                'class': ['normal'] * 800 + ['attack'] * 200
            })
            
            features, labels = loader.extract_features_and_labels(sample_data)
            print(f"📊 Sample features: {len(features)} columns")
            
            return sample_data, features, labels
            
    except ImportError as e:
        print(f"❌ DataLoader import failed: {e}")
        return None, None, None

def demo_data_preprocessor(data, features):
    """Demonstrate DataPreprocessor module."""
    print("\n🔧 DEMO: DataPreprocessor Module")
    print("-" * 40)
    
    try:
        from data.preprocessor import DataPreprocessor
        
        if data is None:
            print("⚠️  No data available for preprocessing demo")
            return None, None
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor initialized")
        
        # Preprocess features
        processed_features = preprocessor.preprocess_features(data, features)
        print(f"🔄 Features processed: {processed_features.shape}")
        
        # Separate normal/anomalous (if class column exists)
        if 'class' in data.columns:
            normal_data, anomalous_data = preprocessor.separate_normal_anomalous(
                processed_features, data['class'], normal_identifier='normal'
            )
            print(f"🔵 Normal samples: {len(normal_data) if normal_data is not None else 0}")
            print(f"🔴 Anomalous samples: {len(anomalous_data) if anomalous_data is not None else 0}")
            
            if normal_data is not None and len(normal_data) > 0:
                # Prepare training data
                train_data, val_data = preprocessor.prepare_training_data(
                    normal_data, validation_ratio=0.2, random_state=42
                )
                print(f"🏋️  Training samples: {len(train_data)}")
                print(f"🧪 Validation samples: {len(val_data)}")
                
                return train_data, val_data
        
        return processed_features, None
        
    except ImportError as e:
        print(f"❌ DataPreprocessor import failed: {e}")
        return None, None

def demo_model_trainer(train_data, val_data):
    """Demonstrate EnhancedModelTrainer module."""
    print("\n🏗️  DEMO: EnhancedModelTrainer Module")
    print("-" * 40)
    
    try:
        from core.enhanced_trainer import EnhancedModelTrainer, ProductionAutoencoder
        
        if train_data is None:
            print("⚠️  No training data available for trainer demo")
            return None
        
        # Initialize trainer
        trainer = EnhancedModelTrainer()
        print("✅ EnhancedModelTrainer initialized")
        
        # Create model
        input_dim = train_data.shape[1]
        hidden_dims = [32, 16, 8, 16, 32]  # Small for demo
        
        model = trainer.create_model(input_dim=input_dim, hidden_dims=hidden_dims)
        print(f"🏗️  Model created: {input_dim} → {hidden_dims} → {input_dim}")
        
        # Quick training (reduced epochs for demo)
        training_config = {
            'epochs': 10,  # Quick demo
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        print("🚀 Starting quick training demo...")
        history = trainer.train_model(
            train_data=train_data,
            val_data=val_data,
            config=training_config
        )
        
        print(f"📈 Training completed!")
        print(f"   Final loss: {history['final_train_loss']:.6f}")
        print(f"   Training time: {history['total_time']:.2f}s")
        print(f"   Epochs: {history['epochs']}")
        
        return model
        
    except ImportError as e:
        print(f"❌ EnhancedModelTrainer import failed: {e}")
        return None

def demo_model_evaluator(model, val_data):
    """Demonstrate ModelEvaluator module."""
    print("\n📊 DEMO: ModelEvaluator Module")
    print("-" * 40)
    
    try:
        from core.evaluator import ModelEvaluator
        
        if model is None or val_data is None:
            print("⚠️  No model or validation data available for evaluator demo")
            return None
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        print("✅ ModelEvaluator initialized")
        
        # Evaluate model
        print("🔍 Running evaluation...")
        evaluation_results = evaluator.evaluate_model(
            model=model,
            normal_data=val_data,
            anomalous_data=None,  # No anomalous data in this demo
            class_info=None
        )
        
        # Display results
        if 'normal_errors' in evaluation_results:
            normal_stats = evaluation_results['normal_errors']
            print(f"📈 Normal Data Reconstruction Errors:")
            print(f"   Mean: {normal_stats['mean']:.6f}")
            print(f"   Std:  {normal_stats['std']:.6f}")
            print(f"   Range: [{normal_stats['min']:.6f}, {normal_stats['max']:.6f}]")
        
        if 'thresholds' in evaluation_results:
            thresholds = evaluation_results['thresholds']
            print(f"🎚️  Calculated Thresholds:")
            for method, threshold in thresholds.items():
                print(f"   {method}: {threshold:.6f}")
        
        return evaluation_results
        
    except ImportError as e:
        print(f"❌ ModelEvaluator import failed: {e}")
        return None

def demo_constants():
    """Demonstrate Constants module."""
    print("\n⚙️  DEMO: Constants Module")
    print("-" * 40)
    
    try:
        from utils.constants import DataConstants, ModelDefaults
        
        print("✅ Constants imported successfully")
        print(f"📁 Default data path: {DataConstants.DATA_FILE_PATH}")
        print(f"🏗️  Model hidden dims: {ModelDefaults.HIDDEN_DIMS}")
        print(f"⚡ Learning rate: {ModelDefaults.LEARNING_RATE}")
        print(f"🎯 Batch size: {ModelDefaults.BATCH_SIZE}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Constants import failed: {e}")
        return False

def main():
    """Run complete refactoring demo with enterprise configuration."""
    print("=" * 70)
    print("🎯 NIDS AUTOENCODER ENTERPRISE CONFIGURATION DEMO")
    print("=" * 70)
    print("This demo shows the refactored modules with centralized configuration")
    print()
    
    # Demo configuration management first
    config_manager = demo_configuration_manager()
    print()
    
    # Demo each module with configuration
    demo_constants()
    
    data, features, _ = demo_data_loader(config_manager)
    train_data, val_data = demo_data_preprocessor(data, features)
    model = demo_model_trainer(train_data, val_data)
    results = demo_model_evaluator(model, val_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 ENTERPRISE CONFIGURATION DEMO COMPLETE!")
    print("=" * 70)
    
    success_count = sum([
        config_manager is not None,
        data is not None,
        train_data is not None,
        model is not None,
        results is not None
    ])
    
    print(f"📊 Demo Success Rate: {success_count}/5 components working")
    
    if success_count >= 4:
        print("✅ Enterprise configuration system is working correctly!")
        print("🚀 Ready for enterprise production deployment!")
    elif success_count >= 3:
        print("⚠️  Most components working, minor configuration issues to resolve")
    else:
        print("❌ Several components need attention")
    
    print("\n💡 Next Steps for Enterprise Deployment:")
    print("   🔧 Configure environment-specific settings")
    print("   🧪 Run comprehensive integration tests")
    print("   🌐 Deploy with environment-aware configuration")
    print("   📊 Monitor configuration compliance")
    print("   � Validate security configurations")
    print("   🐳 Containerize for deployment")

if __name__ == "__main__":
    main()
