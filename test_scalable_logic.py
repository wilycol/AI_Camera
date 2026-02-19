import numpy as np
import sys

def test_normalization_logic():
    # Simulate an embedding from the model
    raw_embedding = np.random.rand(512).astype("float32")
    
    # Normalization (as in utils.py)
    norm = np.linalg.norm(raw_embedding)
    normalized = raw_embedding / norm
    
    # Verify norm is 1.0 (or very close)
    new_norm = np.linalg.norm(normalized)
    print(f"Testing Normalization: Raw Norm={norm:.4f}, Normalized Norm={new_norm:.4f}")
    
    assert np.isclose(new_norm, 1.0, atol=1e-5), "Normalization failed!"
    print("✅ Normalization Logic Passed.")
    return normalized

def test_cosine_similarity():
    # Create two identical normalized vectors
    v1 = np.random.rand(512).astype("float32")
    v1 = v1 / np.linalg.norm(v1)
    
    # Identical vector should have dot product 1.0
    score_identical = np.dot(v1, v1)
    print(f"Testing Cosine: Identical Score={score_identical:.4f}")
    assert np.isclose(score_identical, 1.0, atol=1e-5), "Cosine Similarity Identical failed!"
    
    # Different vector should have lower score
    v2 = np.random.rand(512).astype("float32")
    v2 = v2 / np.linalg.norm(v2)
    score_diff = np.dot(v1, v2)
    print(f"Testing Cosine: Different Score={score_diff:.4f}")
    assert score_diff < 1.0, "Cosine Similarity Different failed!"
    
    print("✅ Cosine Similarity Logic Passed.")

if __name__ == "__main__":
    print("--- AI_Camera Scalable Logic Verification ---")
    try:
        test_normalization_logic()
        test_cosine_similarity()
        print("--- ALL TESTS PASSED ---")
    except Exception as e:
        print(f"--- TEST FAILED: {e} ---")
        sys.exit(1)
