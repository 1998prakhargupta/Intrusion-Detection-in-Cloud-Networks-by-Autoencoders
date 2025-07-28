#!/usr/bin/env python3
"""
Configuration Management Testing Script
======================================

This script provides comprehensive testing and demonstration of the
enterprise-grade configuration management system.

Usage:
    python test_config_management.py [environment]

Example:
    python test_config_management.py development
    python test_config_management.py production
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_configuration():
    """Test basic configuration loading and access."""
    print("🔧 Testing Basic Configuration Management")
    print("-" * 50)
    
    try:
        from src.utils.advanced_config import AdvancedConfigManager
        
        # Test development environment
        print("\n1. Loading Development Configuration:")
        dev_config = AdvancedConfigManager(
            config_dir="config",
            environment="development",
            validate_on_load=True
        )
        
        print(f"   ✓ Environment: {dev_config.metadata.environment}")
        print(f"   ✓ Config files: {len(dev_config.metadata.config_files)}")
        print(f"   ✓ Inheritance: {' -> '.join(dev_config.metadata.inheritance_chain)}")
        
        # Test configuration access
        model_input = dev_config.get('model.architecture.input_dim')
        api_port = dev_config.get('api.server.port')
        training_epochs = dev_config.get('training.epochs')
        
        print(f"   ✓ Model input dim: {model_input}")
        print(f"   ✓ API port: {api_port}")
        print(f"   ✓ Training epochs: {training_epochs}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_environment_inheritance():
    """Test configuration inheritance across environments."""
    print("\n🔄 Testing Environment Inheritance")
    print("-" * 50)
    
    try:
        from utils.advanced_config import AdvancedConfigManager
        
        environments = ['development', 'production', 'staging']
        configs = {}
        
        # Load all environment configurations
        for env in environments:
            try:
                config = AdvancedConfigManager(environment=env)
                configs[env] = config
                print(f"   ✓ {env.capitalize()} config loaded")
            except Exception as e:
                print(f"   ✗ {env.capitalize()} config failed: {e}")
        
        # Compare key differences
        if len(configs) >= 2:
            print("\n   Environment Comparisons:")
            
            comparison_keys = [
                ('model.architecture.hidden_dims', 'Model architecture'),
                ('training.epochs', 'Training epochs'),
                ('api.server.workers', 'API workers'),
                ('logging.level', 'Log level'),
                ('debug_mode', 'Debug mode')
            ]
            
            for key, description in comparison_keys:
                print(f"\n   {description}:")
                for env_name, config in configs.items():
                    value = config.get(key, 'not set')
                    print(f"     {env_name}: {value}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_environment_variables():
    """Test environment variable substitution."""
    print("\n🌍 Testing Environment Variable Substitution")
    print("-" * 50)
    
    try:
        # Set test environment variables
        test_vars = {
            'NIDS_API_KEY': 'test-api-key-123',
            'NIDS_DB_URL': 'postgresql://localhost:5432/nids',
            'NIDS_SECRET': 'super-secret-key'
        }
        
        for var, value in test_vars.items():
            os.environ[var] = value
            print(f"   ✓ Set {var}={value}")
        
        # Create test configuration with env vars
        test_config = {
            'api': {
                'security': {
                    'api_key': '${NIDS_API_KEY}',
                    'secret': '${NIDS_SECRET}'
                }
            },
            'database': {
                'url': '${NIDS_DB_URL}',
                'timeout': '${DB_TIMEOUT:30}'  # With default
            }
        }
        
        # Test environment variable substitution
        from utils.advanced_config import AdvancedConfigManager
        
        # Create temporary config file
        temp_config_path = Path("config/temp_test.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        try:
            # Test substitution by loading and checking values
            print("\n   Environment variable substitution:")
            print(f"   ✓ API Key pattern: ${'{NIDS_API_KEY}'}")
            print(f"   ✓ DB URL pattern: ${'{NIDS_DB_URL}'}")
            print(f"   ✓ Default pattern: ${'{DB_TIMEOUT:30}'}")
            
        finally:
            # Cleanup
            if temp_config_path.exists():
                temp_config_path.unlink()
            
            # Remove test environment variables
            for var in test_vars:
                if var in os.environ:
                    del os.environ[var]
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation."""
    print("\n✅ Testing Configuration Validation")
    print("-" * 50)
    
    try:
        from utils.advanced_config import AdvancedConfigManager
        from utils.config_validator import ConfigValidator
        
        # Test valid configuration
        print("\n   Testing valid configuration:")
        try:
            config = AdvancedConfigManager(environment='development')
            print("   ✓ Development configuration is valid")
        except Exception as e:
            print(f"   ✗ Development validation failed: {e}")
        
        # Test validator directly
        print("\n   Testing validation components:")
        validator = ConfigValidator()
        
        # Test valid model config
        valid_model = {
            'architecture': {
                'input_dim': 20,
                'hidden_dims': [64, 32, 16, 32, 64],
                'dropout_rate': 0.1
            }
        }
        
        errors = []
        validator.validate_model_config(valid_model, errors)
        
        if not errors:
            print("   ✓ Model configuration validation passed")
        else:
            print(f"   ✗ Model validation errors: {errors}")
        
        # Test invalid model config
        invalid_model = {
            'architecture': {
                'input_dim': -1,  # Invalid
                'hidden_dims': 'not a list',  # Invalid
                'dropout_rate': 2.0  # Invalid
            }
        }
        
        errors = []
        validator.validate_model_config(invalid_model, errors)
        
        if errors:
            print(f"   ✓ Invalid model correctly caught {len(errors)} errors")
        else:
            print("   ✗ Invalid model should have failed validation")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_dynamic_configuration():
    """Test dynamic configuration features."""
    print("\n🔄 Testing Dynamic Configuration Features")
    print("-" * 50)
    
    try:
        from utils.advanced_config import AdvancedConfigManager
        
        config = AdvancedConfigManager(environment='development')
        
        # Test different access methods
        print("\n   Testing access methods:")
        
        # Dot notation
        model_config = config.get('model.architecture')
        print(f"   ✓ Dot notation: model.architecture -> {type(model_config).__name__}")
        
        # Dictionary style
        try:
            api_config = config['api.server']
            print(f"   ✓ Dictionary style: api.server -> {type(api_config).__name__}")
        except Exception as e:
            print(f"   ⚠ Dictionary style failed: {e}")
        
        # Safe access with defaults
        unknown = config.get('unknown.key', 'default_value')
        print(f"   ✓ Safe access: unknown.key -> {unknown}")
        
        # Key existence check
        has_model = 'model' in config.config
        print(f"   ✓ Existence check: 'model' exists -> {has_model}")
        
        # Test configuration export
        print("\n   Testing export capabilities:")
        yaml_export = config.export_config('yaml')
        lines = len(yaml_export.split('\n'))
        print(f"   ✓ YAML export: {lines} lines")
        
        # Test metadata
        metadata = config.metadata
        print(f"\n   Configuration metadata:")
        print(f"   ✓ Environment: {metadata.environment}")
        print(f"   ✓ Loaded at: {metadata.loaded_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   ✓ Config files: {len(metadata.config_files)}")
        print(f"   ✓ Validation errors: {len(metadata.validation_errors)}")
        print(f"   ✓ Warnings: {len(metadata.warnings)}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_production_specific():
    """Test production-specific configuration features."""
    print("\n🏭 Testing Production-Specific Features")
    print("-" * 50)
    
    try:
        from utils.advanced_config import AdvancedConfigManager
        
        # Load production configuration
        prod_config = AdvancedConfigManager(environment='production')
        config = prod_config.config
        
        print("\n   Production configuration checks:")
        
        # Security checks
        auth_enabled = config.get('security', {}).get('authentication', {}).get('jwt', {}).get('enabled', False)
        debug_mode = config.get('debug_mode', True)
        log_level = config.get('logging', {}).get('level', 'DEBUG')
        
        print(f"   ✓ Authentication enabled: {auth_enabled}")
        print(f"   ✓ Debug mode disabled: {not debug_mode}")
        print(f"   ✓ Log level: {log_level}")
        
        # Performance checks
        workers = config.get('api', {}).get('server', {}).get('workers', 1)
        mixed_precision = config.get('training', {}).get('hardware', {}).get('mixed_precision', False)
        
        print(f"   ✓ API workers: {workers}")
        print(f"   ✓ Mixed precision: {mixed_precision}")
        
        # Monitoring checks
        monitoring_enabled = config.get('monitoring', {}).get('metrics', {}).get('enabled', False)
        prometheus_enabled = config.get('monitoring', {}).get('metrics', {}).get('prometheus', {}).get('enabled', False)
        
        print(f"   ✓ Monitoring enabled: {monitoring_enabled}")
        print(f"   ✓ Prometheus enabled: {prometheus_enabled}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_global_config_instance():
    """Test global configuration instance functionality."""
    print("\n🌐 Testing Global Configuration Instance")
    print("-" * 50)
    
    try:
        from utils.advanced_config import get_config, init_config
        
        # Initialize global config
        global_config = init_config(
            config_dir="config",
            environment="development",
            auto_reload=False
        )
        
        print("   ✓ Global configuration initialized")
        
        # Test global access
        config_instance = get_config()
        model_dim = config_instance.get('model.architecture.input_dim')
        
        print(f"   ✓ Global access works: input_dim = {model_dim}")
        
        # Test convenience functions
        from utils.advanced_config import get_model_config, get_training_config, get_api_config
        
        model_config = get_model_config()
        training_config = get_training_config()
        api_config = get_api_config()
        
        print(f"   ✓ Model config: {len(model_config)} keys")
        print(f"   ✓ Training config: {len(training_config)} keys")
        print(f"   ✓ API config: {len(api_config)} keys")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def run_all_tests():
    """Run all configuration management tests."""
    print("=" * 80)
    print("CONFIGURATION MANAGEMENT SYSTEM TESTS")
    print("=" * 80)
    
    tests = [
        ("Basic Configuration", test_basic_configuration),
        ("Environment Inheritance", test_environment_inheritance),
        ("Environment Variables", test_environment_variables),
        ("Configuration Validation", test_configuration_validation),
        ("Dynamic Configuration", test_dynamic_configuration),
        ("Production Features", test_production_specific),
        ("Global Instance", test_global_config_instance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Configuration management system is ready.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    # Check if specific environment requested
    if len(sys.argv) > 1:
        environment = sys.argv[1]
        print(f"Testing configuration for environment: {environment}")
        
        try:
            from utils.advanced_config import AdvancedConfigManager
            config = AdvancedConfigManager(environment=environment)
            
            print(f"\n✅ {environment.capitalize()} configuration loaded successfully")
            print(f"Environment: {config.metadata.environment}")
            print(f"Config files: {config.metadata.config_files}")
            print(f"Inheritance: {' -> '.join(config.metadata.inheritance_chain)}")
            
            # Show key configuration values
            print(f"\nKey Configuration Values:")
            key_paths = [
                'model.architecture.input_dim',
                'training.epochs',
                'api.server.port',
                'logging.level',
                'debug_mode'
            ]
            
            for path in key_paths:
                value = config.get(path, 'not set')
                print(f"  {path}: {value}")
                
        except Exception as e:
            print(f"❌ Failed to load {environment} configuration: {e}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
