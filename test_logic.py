import sys
import os

# Mocking libraries that might not be installed yet for local verification of logic
try:
    import cv2
    import numpy as np
    from mtcnn import MTCNN
except ImportError:
    print("Warning: OpenCV, NumPy or MTCNN not installed. This script requires dependencies to run fully.")

def test_imports():
    try:
        import utils
        import db_config
        import server
        print("✅ Core modules imported successfully.")
        return True
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False

def test_utils_logic():
    # Test if utils functions are callable and defined
    required_funcs = [
        'load_model', 'get_face', 'forward_pass', 
        'save_embedding', 'load_embeddings', 'identify_face', 'log_event'
    ]
    import utils
    missing = [f for f in required_funcs if not hasattr(utils, f)]
    if not missing:
        print("✅ All required utility functions are present.")
        return True
    else:
        print(f"❌ Missing utility functions: {missing}")
        return False

if __name__ == "__main__":
    print("--- AI_Camera Logic Verification ---")
    if test_imports() and test_utils_logic():
        print("--- Verification PASSED (Logic Only) ---")
        print("Note: Full functional test requires 'facenet_keras.h5' and MongoDB.")
    else:
        print("--- Verification FAILED ---")
        sys.exit(1)
