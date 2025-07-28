"""Test minimal imports."""

import sys
sys.path.insert(0, '.')

# Test utils imports individually
try:
    import src.utils.constants
    print("✓ Constants module imported")
    print("  THRESHOLD_PERCENTILE:", src.utils.constants.ModelDefaults.THRESHOLD_PERCENTILE)
except Exception as e:
    print("✗ Constants import failed:", e)
    import traceback
    traceback.print_exc()

try:
    import src.utils.data_utils
    print("✓ Data utils module imported")
except Exception as e:
    print("✗ Data utils import failed:", e)
    import traceback
    traceback.print_exc()

try:
    import src.utils.api_utils
    print("✓ API utils module imported")
except Exception as e:
    print("✗ API utils import failed:", e)
    import traceback
    traceback.print_exc()
