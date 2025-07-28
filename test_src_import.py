#!/usr/bin/env python3
"""Minimal test to debug src package import."""

import sys
import time
sys.path.insert(0, '.')

print("Starting import test...")
start_time = time.time()

try:
    print("Step 1: Testing basic src module import")
    import src
    print(f"✓ src imported successfully in {time.time() - start_time:.2f}s")
    print(f"Available attributes: {dir(src)}")
    
    if hasattr(src, '__all__'):
        print(f"Available modules: {src.__all__}")
    
except Exception as e:
    print(f"✗ src import failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
