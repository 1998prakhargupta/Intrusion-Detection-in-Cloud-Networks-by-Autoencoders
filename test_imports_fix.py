#!/usr/bin/env python3
"""
Test script to verify the import fix
"""

def test_imports():
    """Test if the import errors are fixed."""
    
    print("Testing import fixes...")
    
    try:
        # Test 1: Direct config.py import
        print("\n1. Testing config.py import...")
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from src.utils.config import Config
        print("   ✅ config.py import successful")
        
    except Exception as e:
        print(f"   ❌ config.py import failed: {e}")
    
    try:
        # Test 2: config_manager import
        print("\n2. Testing config_manager import...")
        from src.utils.config_manager import SimpleConfigManager
        
        # Create instance
        config = SimpleConfigManager(environment='development')
        print("   ✅ SimpleConfigManager created successfully")
        
        # Test configuration access
        model_dim = config.get('model.architecture.input_dim')
        print(f"   ✅ Configuration access working: input_dim = {model_dim}")
        
    except Exception as e:
        print(f"   ❌ config_manager failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test 3: utils __init__.py import
        print("\n3. Testing utils package import...")
        from src.utils import Config, load_config, get_env_var
        print("   ✅ utils package imports successful")
        
        # Test functions
        env_var = get_env_var('HOME', 'default')
        print(f"   ✅ get_env_var working: HOME exists = {env_var is not None}")
        
        config_instance = load_config(environment='development')
        print(f"   ✅ load_config working: type = {type(config_instance).__name__}")
        
    except Exception as e:
        print(f"   ❌ utils package import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("IMPORT TEST COMPLETED")
    print("="*50)

if __name__ == "__main__":
    test_imports()
