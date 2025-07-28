#!/usr/bin/env python3
"""Simple test to import src package step by step."""

import sys
sys.path.insert(0, '.')

print("Testing individual module imports:")

# Test 1: Basic utils
try:
    from src.utils.constants import ModelDefaults
    print("✓ Constants imported")
except ImportError as e:
    print(f"✗ Constants import failed: {e}")

# Test 2: Models
try:
    from src.models.autoencoder import AutoencoderModel
    print("✓ AutoencoderModel imported")
except ImportError as e:
    print(f"✗ AutoencoderModel import failed: {e}")

# Test 3: Core modules
try:
    from src.core.trainer import ModelTrainer
    print("✓ ModelTrainer imported")
except ImportError as e:
    print(f"✗ ModelTrainer import failed: {e}")

try:
    from src.core.predictor import AnomalyPredictor
    print("✓ AnomalyPredictor imported")
except ImportError as e:
    print(f"✗ AnomalyPredictor import failed: {e}")

# Test 4: Data modules
try:
    from src.data.processor import DataProcessor
    print("✓ DataProcessor imported")
except ImportError as e:
    print(f"✗ DataProcessor import failed: {e}")

# Test 5: API (this might be the problem)
try:
    from src.api.main import app
    print("✓ API app imported")
except ImportError as e:
    print(f"✗ API app import failed: {e}")

print("\nAll individual imports completed!")
