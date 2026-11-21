import pytest
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.services.fuzzy_engine import FuzzyReasoningEngine
from app.services.uncertainty import UncertaintyAssessmentService

def test_fuzzy_reasoning_engine():
    engine = FuzzyReasoningEngine()
    
    # High confidence, good segmentation, consistent features -> Low uncertainty
    res_low = engine.assess_uncertainty(
        prediction_confidence=0.92,
        segmentation_quality=0.88,
        boundary_clarity=0.85,
        feature_consistency=0.90
    )
    assert res_low["fuzzy_uncertainty_score"] < 0.40
    assert res_low["fuzzy_uncertainty_level"] == "Low Uncertainty"
    assert res_low["abstain_recommended"] is False

    # Low confidence -> High uncertainty and abstain
    res_high = engine.assess_uncertainty(
        prediction_confidence=0.35,
        segmentation_quality=0.30,
        boundary_clarity=0.20,
        feature_consistency=0.40
    )
    assert res_high["fuzzy_uncertainty_score"] > 0.60
    assert res_high["fuzzy_uncertainty_level"] == "High Uncertainty"
    assert res_high["needs_expert_review"] is True

def test_uncertainty_assessment_service():
    service = UncertaintyAssessmentService()
    
    cls_result = {
        "confidence_score": 0.88,
        "class_probabilities": {"Glioma": 0.88, "Meningioma": 0.06, "Pituitary": 0.04, "No Tumor": 0.02},
        "mc_sample_matrix": np.tile([0.88, 0.06, 0.04, 0.02], (15, 1)).tolist(),
        "predicted_class": "Glioma"
    }
    seg_result = {"tumor_area_pixels": 1500}
    extracted_feats = {
        "shape_features": {"area_pixels": 1500, "circularity": 0.45, "boundary_irregularity": 0.55, "solidity": 0.85},
        "boundary_characteristics": {"clarity_score": 0.70},
        "intensity_features": {"mean_intensity": 160.0, "std_intensity": 50.0},
        "texture_features": {"glcm_contrast": 35.0, "glcm_homogeneity": 0.35}
    }
    
    unc_res = service.assess(cls_result, seg_result, extracted_feats, modality="MRI")
    
    assert "predictive_entropy" in unc_res
    assert "epistemic_uncertainty" in unc_res
    assert "fuzzy_uncertainty_level" in unc_res
    assert unc_res["confidence_score"] == 0.88
    assert unc_res["predictive_entropy"] >= 0.0
