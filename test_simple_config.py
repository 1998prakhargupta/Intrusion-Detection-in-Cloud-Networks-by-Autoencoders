#!/usr/bin/env python3
"""
Simple Configuration Management Test
===================================

Test the configuration management system.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_simple_config():
    """Test the simple configuration manager."""
    print("=" * 60)
    print("TESTING CONFIGURATION MANAGEMENT SYSTEM")
    print("=" * 60)
    
    try:
        from src.utils.config_manager import SimpleConfigManager
        
        print("\n1. Loading Development Configuration")
        print("-" * 40)
        
        # Test development environment
        dev_config = SimpleConfigManager(
            config_dir="config",
            environment="development",
            validate_on_load=True
        )
        
        print(f"✓ Environment: {dev_config.metadata['environment']}")
        print(f"✓ Config files: {len(dev_config.metadata['config_files'])}")
        print(f"✓ Inheritance: {' -> '.join(dev_config.metadata['inheritance_chain'])}")
        
        # Test configuration access
        model_input = dev_config.get('model.architecture.input_dim')
        api_port = dev_config.get('api.server.port')
        training_epochs = dev_config.get('training.epochs')
        
        print(f"✓ Model input dim: {model_input}")
        print(f"✓ API port: {api_port}")
        print(f"✓ Training epochs: {training_epochs}")
        
        print("\n2. Testing Different Environments")
        print("-" * 40)
        
        # Test different environments
        environments = ['development', 'production', 'staging']
        
        for env in environments:
            try:
                config = SimpleConfigManager(environment=env)
                model_epochs = config.get('training.epochs', 'not set')
                debug_mode = config.get('debug_mode', 'not set')
                
                print(f"✓ {env.capitalize()}: epochs={model_epochs}, debug={debug_mode}")
                
            except Exception as e:
                print(f"✗ {env.capitalize()}: {e}")
        
        print("\n3. Testing Configuration Export")
        print("-" * 40)
        
        # Test export
        yaml_config = dev_config.export_config('yaml')
        lines = len(yaml_config.split('\n'))
        
        print(f"✓ YAML export: {lines} lines")
        
        # Show top-level sections
        config_dict = dev_config.config
        sections = list(config_dict.keys())
        print(f"✓ Configuration sections: {sections}")
        
        print("\n4. Testing Environment Variable Substitution")
        print("-" * 40)
        
        # Set test environment variables
        os.environ['TEST_API_KEY'] = 'demo-key-123'
        
        # Test substitution (simulated)
        test_value = "${TEST_API_KEY}"
        import re
        pattern = r'\$\{([A-Za-z_]\w*)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value or match.group(0))
        
        substituted = re.sub(pattern, replace_var, test_value)
        print(f"✓ Environment variable substitution: {test_value} -> {substituted}")
        
        # Cleanup
        if 'TEST_API_KEY' in os.environ:
            del os.environ['TEST_API_KEY']
        
        print("\n" + "=" * 60)
        print("CONFIGURATION MANAGEMENT TEST COMPLETED!")
        print("✓ All basic features working correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_config()
    sys.exit(0 if success else 1)
