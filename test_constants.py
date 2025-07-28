#!/usr/bin/env python3
"""Simple test for constants import."""

try:
    print("Testing constants import...")
    from src.utils.constants import APIConstants
    print("✓ APIConstants imported successfully")
    print(f"RATE_LIMIT_REQUESTS: {APIConstants.RATE_LIMIT_REQUESTS}")
    print(f"MAX_BATCH_SIZE: {APIConstants.MAX_BATCH_SIZE}")
    print(f"RATE_LIMIT_WINDOW: {APIConstants.RATE_LIMIT_WINDOW}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
