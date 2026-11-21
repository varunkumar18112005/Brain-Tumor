from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ImageUploadResponse(BaseModel):
    file_id: str
    original_filename: str
    filepath: str
    relative_url: str
    dimensions: List[int]
    format: str
    filesize_kb: float
    modality: str
    patient_id: str
    total_slices: Optional[int] = 1
    selected_slice: Optional[int] = 0

class PreprocessingRequest(BaseModel):
    file_id: str
    modality: str = "MRI"
    gaussian_kernel: int = 3
    median_kernel: int = 3
    clahe_clip_limit: float = 2.0
    target_size: int = 224

class PreprocessingResponse(BaseModel):
    file_id: str
    original_url: str
    denoised_url: str
    enhanced_url: str
    normalized_url: str
    transformations_applied: List[str]

class SegmentationRequest(BaseModel):
    file_id: str
    modality: str = "MRI"
    use_model: bool = True

class SegmentationResponse(BaseModel):
    file_id: str
    segmentation_mask_url: str
    tumor_overlay_url: str
    extracted_region_url: str
    tumor_area_pixels: int
    tumor_perimeter_pixels: float
    circularity: float
    boundary_irregularity: float
    bounding_box: Dict[str, int]
    model_used: str
    is_trained_model: bool
    dice_score: Optional[float] = None
    iou_score: Optional[float] = None

class FeatureExtractionRequest(BaseModel):
    file_id: str

class ExtractedFeaturesResponse(BaseModel):
    file_id: str
    intensity_features: Dict[str, float]
    texture_features: Dict[str, float]
    shape_features: Dict[str, float]
    boundary_characteristics: Dict[str, Any]

class ClassificationRequest(BaseModel):
    file_id: str
    modality: str = "MRI"

class ClassificationResponse(BaseModel):
    file_id: str
    tumor_detected: bool
    predicted_class: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    model_name: str
    is_trained: bool

class UncertaintyRequest(BaseModel):
    file_id: str
    modality: str = "MRI"
    mc_samples: int = 15

class UncertaintyResponse(BaseModel):
    file_id: str
    confidence_score: float
    predictive_entropy: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    fuzzy_uncertainty_score: float
    fuzzy_uncertainty_level: str
    fuzzy_explanation: str
    fired_rules: List[Dict[str, Any]]
    needs_expert_review: bool
    abstain_recommended: bool

class FullAnalysisRequest(BaseModel):
    file_id: str
    patient_id: str
    modality: str = "MRI"
    selected_slice: int = 0

class FullAnalysisResponse(BaseModel):
    analysis_id: int
    patient_id: str
    modality: str
    scan_filename: str
    created_at: str
    
    # Image Visualizations
    original_url: str
    denoised_url: str
    enhanced_url: str
    normalized_url: str
    segmentation_mask_url: str
    tumor_overlay_url: str
    extracted_region_url: str
    
    # Tumor Findings
    tumor_detected: bool
    predicted_class: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    
    # Features
    extracted_features: Dict[str, Any]
    
    # Uncertainty
    predictive_entropy: float
    epistemic_uncertainty: float
    fuzzy_uncertainty_score: float
    fuzzy_uncertainty_level: str
    fuzzy_reasoning_explanation: str
    fired_fuzzy_rules: List[Dict[str, Any]]
    
    # Decision Support
    needs_expert_review: bool
    abstain_recommended: bool
    decision_summary: str
    report_pdf_url: Optional[str] = None
    model_status: Dict[str, Any]

class ModelStatusResponse(BaseModel):
    models: List[Dict[str, Any]]
    active_device: str
