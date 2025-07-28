#!/usr/bin/env python3
"""Systematic error detection test."""

import sys
import traceback
sys.path.insert(0, '.')

def test_import(module_name, description):
    """Test a specific import and report errors."""
    try:
        if module_name == "src":
            import src
        elif module_name == "src.utils":
            import src.utils
        elif module_name == "src.utils.config_utils":
            from src.utils.config_utils import ConfigManager, DataConfig
        elif module_name == "src.utils.constants":
            from src.utils.constants import ModelDefaults, APIConstants, DataConstants
        elif module_name == "src.models":
            from src.models import AutoencoderModel
        elif module_name == "src.utils.logger":
            from src.utils.logger import get_logger
        
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        traceback.print_exc()
        print("-" * 50)
        return False

# Test imports step by step
tests = [
    ("src.utils.constants", "Constants import"),
    ("src.utils.logger", "Logger import"), 
    ("src.utils.config_utils", "Config utils import"),
    ("src.utils", "Utils package import"),
    ("src.models", "Models import"),
    ("src", "Main src package import"),
]

failed_tests = []
for module, desc in tests:
    if not test_import(module, desc):
        failed_tests.append(desc)

if failed_tests:
    print(f"\n❌ {len(failed_tests)} tests failed:")
    for test in failed_tests:
        print(f"  - {test}")
else:
    print("\n🎉 All imports successful!")
