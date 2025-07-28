import sys
print("Testing type annotation fix...")
print("Python version:", sys.version)

sys.path.insert(0, 'src/utils')

try:
    from config import DataConfig, ThresholdConfig
    print("✅ SUCCESS: Type annotations fixed!")
    
    # Test the classes
    dc = DataConfig()
    tc = ThresholdConfig()
    
    print(f"DataConfig feature_range: {dc.feature_range}")
    print(f"DataConfig selected_features: {dc.selected_features}")
    print(f"ThresholdConfig methods: {tc.methods}")
    
    print("🎉 ALL TYPE ANNOTATION ERRORS RESOLVED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    if "'type' object is not subscriptable" in str(e):
        print("Type annotation error still exists")
    else:
        print("Different error occurred")
