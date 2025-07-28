#!/usr/bin/env python3
"""Test syntax of specific imports."""

import sys
import traceback

# Test individual imports
tests = [
    ("Basic pandas/numpy", "import pandas as pd; import numpy as np"),
    ("Typing imports", "from typing import Union, List, Tuple, Optional, Dict, Any"),
    ("Constants import", "from src.utils.constants import ModelDefaults"),
    ("Data utils import", "from src.utils.data_utils import normalize_features, DataValidator"),
    ("Logger import", "from src.utils.logger import get_logger"),
]

for test_name, test_code in tests:
    try:
        exec(test_code)
        print(f"✓ {test_name}: OK")
    except Exception as e:
        print(f"✗ {test_name}: {e}")
        traceback.print_exc()
        print()
