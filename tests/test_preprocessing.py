import pytest
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.services.preprocessing import PreprocessingService

def test_preprocessing_transformations():
    # Create test synthetic image
    img = np.random.uniform(0.1, 0.9, (128, 128)).astype(np.float32)
    file_id = "test_prep_001"
    
    res = PreprocessingService.preprocess_scan(
        file_id=file_id,
        slice_array=img,
        gaussian_kernel=3,
        median_kernel=3,
        clahe_clip_limit=2.0,
        target_size=224
    )
    
    assert res["file_id"] == file_id
    assert "denoised_array" in res
    assert "enhanced_array" in res
    assert "normalized_array" in res
    assert res["normalized_array"].shape == (224, 224)
    assert 0.0 <= np.min(res["normalized_array"])
    assert np.max(res["normalized_array"]) <= 1.0
    assert len(res["transformations_applied"]) >= 4
