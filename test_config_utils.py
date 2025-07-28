#!/usr/bin/env python3
"""Test config utils import specifically."""

import sys
sys.path.insert(0, '.')

try:
    print("Testing config_utils import...")
    from src.utils.config_utils import DataConfig, ModelConfig
    print("✓ Config utils imported successfully")
except Exception as e:
    print(f"✗ Config utils import failed: {e}")
    import traceback
    traceback.print_exc()
