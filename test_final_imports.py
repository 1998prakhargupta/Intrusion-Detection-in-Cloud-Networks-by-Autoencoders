#!/usr/bin/env python3
"""Final comprehensive import test."""

import sys
sys.path.insert(0, '.')

print("=== NIDS Project Import Test ===")

# Test 1: Constants
try:
    from src.utils.constants import ModelDefaults, DataConstants, APIConstants, PerformanceConstants
    print("✓ Constants imported successfully")
    print(f"  - THRESHOLD_PERCENTILE: {ModelDefaults.THRESHOLD_PERCENTILE}")
    print(f"  - METRICS_HISTORY_SIZE: {PerformanceConstants.METRICS_HISTORY_SIZE}")
    print(f"  - RATE_LIMIT_REQUESTS: {APIConstants.RATE_LIMIT_REQUESTS}")
    print(f"  - TEST_RATIO: {DataConstants.TEST_RATIO}")
except Exception as e:
    print(f"✗ Constants import failed: {e}")
    sys.exit(1)

# Test 2: Data utilities
try:
    from src.utils.data_utils import normalize_features, DataValidator
    print("✓ Data utilities imported successfully")
except Exception as e:
    print(f"✗ Data utilities import failed: {e}")
    sys.exit(1)

# Test 3: Config utilities  
try:
    from src.utils.config_utils import DataConfig, ModelConfig, APIConfig
    print("✓ Config utilities imported successfully")
except Exception as e:
    print(f"✗ Config utilities import failed: {e}")
    sys.exit(1)

# Test 4: API utilities
try:
    from src.utils.api_utils import RequestValidator, ResponseFormatter
    print("✓ API utilities imported successfully")
except Exception as e:
    print(f"✗ API utilities import failed: {e}")
    sys.exit(1)

# Test 5: Complete utils import
try:
    from src.utils import get_logger, DataValidator, normalize_features
    print("✓ Utils package imported successfully")
except Exception as e:
    print(f"✗ Utils package import failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! Project is deployment-ready!")
