import pytest
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.services.feature_extraction import FeatureExtractionService

def test_feature_extraction_outputs():
    # Synthetic scan and circular lesion mask
    scan = np.ones((100, 100), dtype=np.float32) * 120.0
    y, x = np.ogrid[:100, :100]
    mask = ((x - 50) ** 2 + (y - 50) ** 2 <= 15 ** 2).astype(np.uint8)
    scan[mask > 0] = 200.0  # Hyperintense lesion
    
    feats = FeatureExtractionService.extract_all_features(scan, mask)
    
    assert "intensity_features" in feats
    assert "texture_features" in feats
    assert "shape_features" in feats
    assert "boundary_characteristics" in feats
    
    # Check Intensity
    assert feats["intensity_features"]["mean_intensity"] > 180.0
    
    # Check GLCM Texture
    assert "glcm_contrast" in feats["texture_features"]
    assert "glcm_homogeneity" in feats["texture_features"]
    assert 0.0 <= feats["texture_features"]["glcm_homogeneity"] <= 1.0
    
    # Check Shape
    assert feats["shape_features"]["area_pixels"] > 0
    assert feats["shape_features"]["circularity"] > 0.5  # Discrete grid circle
    assert feats["boundary_characteristics"]["lesion_detected"] is True
