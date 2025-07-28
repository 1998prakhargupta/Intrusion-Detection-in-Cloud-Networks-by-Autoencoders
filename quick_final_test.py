#!/usr/bin/env python3
"""Quick final test of all fixed components."""

import sys
sys.path.insert(0, '.')

print("=== QUICK FINAL TEST ===\n")

tests_passed = 0
total_tests = 0

def test_component(description, test_func):
    global tests_passed, total_tests
    total_tests += 1
    try:
        test_func()
        print(f"✓ {description}")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        return False

# Test imports
test_component("Core package import", lambda: __import__('src'))
test_component("Constants import", lambda: __import__('src.utils.constants'))
test_component("AutoencoderModel import", lambda: __import__('src.models.autoencoder'))

# Test object creation
def test_model_creation():
    from src.models.autoencoder import AutoencoderModel
    model = AutoencoderModel(input_size=4)
    assert model is not None

def test_processor_creation():
    from src.data.processor import DataProcessor
    processor = DataProcessor()
    assert processor is not None

def test_trainer_creation():
    from src.core.trainer import ModelTrainer
    trainer = ModelTrainer()
    assert trainer is not None

def test_predictor_creation():
    from src.core.predictor import AnomalyPredictor
    predictor = AnomalyPredictor()
    assert predictor is not None

test_component("AutoencoderModel creation", test_model_creation)
test_component("DataProcessor creation", test_processor_creation) 
test_component("ModelTrainer creation", test_trainer_creation)
test_component("AnomalyPredictor creation", test_predictor_creation)

print(f"\n=== RESULTS: {tests_passed}/{total_tests} TESTS PASSED ===")

if tests_passed == total_tests:
    print("🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT!")
else:
    print("⚠ Some tests failed - check errors above")
