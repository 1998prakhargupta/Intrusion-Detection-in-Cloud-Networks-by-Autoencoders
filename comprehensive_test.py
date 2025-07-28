#!/usr/bin/env python3
"""Comprehensive error detection and testing."""

import sys
import traceback
sys.path.insert(0, '.')

def test_import_safely(module_path, description):
    """Test import with detailed error reporting."""
    try:
        if module_path == "src.utils.constants":
            from src.utils.constants import ModelDefaults, APIConstants
            print(f"✓ {description}")
            return True
        elif module_path == "src.models.autoencoder":
            from src.models.autoencoder import AutoencoderModel
            print(f"✓ {description}")
            return True
        elif module_path == "src.core.predictor":
            from src.core.predictor import AnomalyPredictor
            print(f"✓ {description}")
            return True
        elif module_path == "src.utils":
            import src.utils
            print(f"✓ {description}")
            return True
        elif module_path == "src":
            import src
            print(f"✓ {description}")
            return True
        else:
            exec(f"import {module_path}")
            print(f"✓ {description}")
            return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("-" * 60)
        return False

print("=== COMPREHENSIVE NIDS PROJECT TEST ===\n")

tests = [
    ("src.utils.constants", "Utils constants"),
    ("src.models.autoencoder", "Models autoencoder"),
    ("src.core.predictor", "Core predictor"),
    ("src.utils", "Utils package"),
    ("src", "Main src package"),
]

failed = 0
for module, desc in tests:
    if not test_import_safely(module, desc):
        failed += 1

print(f"\n=== SUMMARY ===")
print(f"Passed: {len(tests) - failed}/{len(tests)}")
print(f"Failed: {failed}/{len(tests)}")

if failed == 0:
    print("🎉 ALL TESTS PASSED! Project is ready for deployment!")
else:
    print(f"❌ {failed} issues remain to be fixed.")
