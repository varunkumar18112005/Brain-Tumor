import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import AnalysisHistory, ModelMetadata
from app.schemas.analysis import (
    ImageUploadResponse, PreprocessingRequest, PreprocessingResponse,
    SegmentationRequest, SegmentationResponse, FeatureExtractionRequest,
    ExtractedFeaturesResponse, ClassificationRequest, ClassificationResponse,
    UncertaintyRequest, UncertaintyResponse, FullAnalysisRequest, FullAnalysisResponse,
    ModelStatusResponse
)
from app.services.acquisition import ImageAcquisitionService
from app.services.preprocessing import PreprocessingService
from app.services.segmentation import TumorSegmentationService
from app.services.feature_extraction import FeatureExtractionService
from app.services.classification import TumorClassificationService
from app.services.uncertainty import UncertaintyAssessmentService
from app.services.decision_support import DecisionSupportService
from app.services.report_generator import ReportGeneratorService
from app.services.knowledge_base import MedicalKnowledgeBaseService
from app.services.fuzzy_engine import FuzzyReasoningEngine
from app.services.model_trainer import ModelTrainingService
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_medical_image(
    file: UploadFile = File(...),
    modality: str = Form("MRI"),
    patient_id: str = Form("")
):
    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        ext = Path(file.filename).suffix.lower()
        if file.filename.endswith(".nii.gz"):
            ext = ".nii.gz"
            
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        res = ImageAcquisitionService.inspect_and_save_upload(
            file_bytes=contents,
            filename=file.filename,
            modality=modality,
            patient_id=patient_id
        )
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image upload and conversion failed: {str(e)}")

@router.post("/preprocess", response_model=PreprocessingResponse)
def preprocess_scan(req: PreprocessingRequest):
    try:
        slice_arr = ImageAcquisitionService.get_slice(req.file_id, 0)
        res = PreprocessingService.preprocess_scan(
            file_id=req.file_id,
            slice_array=slice_arr,
            gaussian_kernel=req.gaussian_kernel,
            median_kernel=req.median_kernel,
            clahe_clip_limit=req.clahe_clip_limit,
            target_size=req.target_size
        )
        return PreprocessingResponse(
            file_id=req.file_id,
            original_url=res["original_url"],
            denoised_url=res["denoised_url"],
            enhanced_url=res["enhanced_url"],
            normalized_url=res["normalized_url"],
            transformations_applied=res["transformations_applied"]
        )
    except Exception as e:
        logger.error(f"Preprocessing failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/segment", response_model=SegmentationResponse)
def segment_scan(req: SegmentationRequest):
    try:
        slice_arr = ImageAcquisitionService.get_slice(req.file_id, 0)
        res = TumorSegmentationService.segment(
            file_id=req.file_id,
            norm_slice=slice_arr,
            modality=req.modality,
            use_deep_model=req.use_model
        )
        return SegmentationResponse(
            file_id=req.file_id,
            segmentation_mask_url=res["segmentation_mask_url"],
            tumor_overlay_url=res["tumor_overlay_url"],
            extracted_region_url=res["extracted_region_url"],
            tumor_area_pixels=res["tumor_area_pixels"],
            tumor_perimeter_pixels=res["tumor_perimeter_pixels"],
            circularity=res["circularity"],
            boundary_irregularity=res["boundary_irregularity"],
            bounding_box=res["bounding_box"],
            model_used=res["model_used"],
            is_trained_model=res["is_trained_model"],
            dice_score=res["dice_score"],
            iou_score=res["iou_score"]
        )
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-features", response_model=ExtractedFeaturesResponse)
def extract_features(req: FeatureExtractionRequest):
    try:
        slice_arr = ImageAcquisitionService.get_slice(req.file_id, 0)
        seg_res = TumorSegmentationService.segment(req.file_id, slice_arr)
        feats = FeatureExtractionService.extract_all_features(slice_arr, seg_res["mask_array"])
        return ExtractedFeaturesResponse(
            file_id=req.file_id,
            intensity_features=feats["intensity_features"],
            texture_features=feats["texture_features"],
            shape_features=feats["shape_features"],
            boundary_characteristics=feats["boundary_characteristics"]
        )
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify", response_model=ClassificationResponse)
def classify_scan(req: ClassificationRequest):
    try:
        slice_arr = ImageAcquisitionService.get_slice(req.file_id, 0)
        seg_res = TumorSegmentationService.segment(req.file_id, slice_arr, modality=req.modality)
        feats = FeatureExtractionService.extract_all_features(slice_arr, seg_res["mask_array"])
        res = TumorClassificationService.classify(
            norm_slice=slice_arr,
            modality=req.modality,
            features=feats
        )
        return ClassificationResponse(
            file_id=req.file_id,
            tumor_detected=res["tumor_detected"],
            predicted_class=res["predicted_class"],
            confidence_score=res["confidence_score"],
            class_probabilities=res["class_probabilities"],
            model_name=res["model_name"],
            is_trained=res["is_trained"]
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uncertainty", response_model=UncertaintyResponse)
def assess_uncertainty(req: UncertaintyRequest):
    try:
        slice_arr = ImageAcquisitionService.get_slice(req.file_id, 0)
        seg_res = TumorSegmentationService.segment(req.file_id, slice_arr, modality=req.modality)
        feats = FeatureExtractionService.extract_all_features(slice_arr, seg_res["mask_array"])
        cls_res = TumorClassificationService.classify(
            norm_slice=slice_arr, modality=req.modality, features=feats, mc_samples=req.mc_samples
        )
        unc_service = UncertaintyAssessmentService()
        unc_res = unc_service.assess(
            classification_result=cls_res,
            segmentation_result=seg_res,
            extracted_features=feats,
            modality=req.modality
        )
        return UncertaintyResponse(
            file_id=req.file_id,
            confidence_score=unc_res["confidence_score"],
            predictive_entropy=unc_res["predictive_entropy"],
            epistemic_uncertainty=unc_res["epistemic_uncertainty"],
            aleatoric_uncertainty=unc_res["aleatoric_uncertainty"],
            fuzzy_uncertainty_score=unc_res["fuzzy_uncertainty_score"],
            fuzzy_uncertainty_level=unc_res["fuzzy_uncertainty_level"],
            fuzzy_explanation=unc_res["fuzzy_explanation"],
            fired_rules=unc_res["fired_fuzzy_rules"],
            needs_expert_review=unc_res["needs_expert_review"],
            abstain_recommended=unc_res["abstain_recommended"]
        )
    except Exception as e:
        logger.error(f"Uncertainty assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=FullAnalysisResponse)
def run_full_analysis(req: FullAnalysisRequest, db: Session = Depends(get_db)):
    """
    Master pipeline endpoint orchestrating Modules 1 through 10:
    1. Acquisition & slice selection
    2. Real Preprocessing
    3. Segmentation & boundary metrics
    4. Radiomic feature extraction
    5. Knowledge base correlation
    6. CNN-based classification with MC Dropout
    7. Fuzzy reasoning & dual uncertainty assessment
    8. Decision support statement
    9. PDF report generation
    10. Storing complete history in SQLite database
    """
    try:
        file_id = req.file_id
        slice_arr = ImageAcquisitionService.get_slice(file_id, req.selected_slice)
        
        # 1. Preprocess
        prep_res = PreprocessingService.preprocess_scan(file_id, slice_arr)
        
        # 2. Segment
        seg_res = TumorSegmentationService.segment(file_id, slice_arr, modality=req.modality)
        
        # 3. Extract Features
        feats = FeatureExtractionService.extract_all_features(slice_arr, seg_res["mask_array"])
        
        # 4. Classify
        cls_res = TumorClassificationService.classify(
            norm_slice=slice_arr, modality=req.modality, features=feats, mc_samples=15
        )
        
        # 5. Assess Uncertainty
        unc_service = UncertaintyAssessmentService()
        unc_res = unc_service.assess(cls_res, seg_res, feats, modality=req.modality)
        
        # 6. Decision Support
        dec_res = DecisionSupportService.generate_clinical_summary(
            patient_id=req.patient_id,
            modality=req.modality,
            classification=cls_res,
            uncertainty=unc_res,
            features=feats,
            is_trained_model=cls_res["is_trained"]
        )

        # 7. Save to Database
        db_record = AnalysisHistory(
            patient_id=req.patient_id,
            modality=req.modality,
            scan_filename=f"scan_{file_id}.png",
            original_filepath=f"/uploads/{file_id}/original.png",
            processed_filepaths={
                "denoised": prep_res["denoised_url"],
                "enhanced": prep_res["enhanced_url"],
                "normalized": prep_res["normalized_url"]
            },
            segmentation_mask_path=seg_res["segmentation_mask_url"],
            tumor_overlay_path=seg_res["tumor_overlay_url"],
            extracted_region_path=seg_res["extracted_region_url"],
            tumor_detected=cls_res["tumor_detected"],
            predicted_class=dec_res["predicted_class"],
            confidence_score=cls_res["confidence_score"],
            class_probabilities=cls_res["class_probabilities"],
            extracted_features=feats,
            predictive_entropy=unc_res["predictive_entropy"],
            epistemic_uncertainty=unc_res["epistemic_uncertainty"],
            aleatoric_uncertainty=unc_res["aleatoric_uncertainty"],
            fuzzy_uncertainty_score=unc_res["fuzzy_uncertainty_score"],
            fuzzy_uncertainty_level=unc_res["fuzzy_uncertainty_level"],
            fuzzy_reasoning_explanation=unc_res["fuzzy_explanation"],
            fired_fuzzy_rules=unc_res["fired_fuzzy_rules"],
            needs_expert_review=dec_res["needs_expert_review"],
            abstain_recommended=dec_res["abstain_recommended"],
            decision_summary=dec_res["decision_summary"]
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        # 8. Generate PDF Report
        report_payload = {
            "analysis_id": db_record.id,
            "patient_id": req.patient_id,
            "modality": req.modality,
            "created_at": db_record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "file_id": file_id,
            "scan_filename": f"scan_{file_id}.png",
            "tumor_detected": cls_res["tumor_detected"],
            "predicted_class": dec_res["predicted_class"],
            "confidence_score": cls_res["confidence_score"],
            "predictive_entropy": unc_res["predictive_entropy"],
            "class_probabilities": cls_res["class_probabilities"],
            "fuzzy_uncertainty_score": unc_res["fuzzy_uncertainty_score"],
            "fuzzy_uncertainty_level": unc_res["fuzzy_uncertainty_level"],
            "fuzzy_reasoning_explanation": unc_res["fuzzy_explanation"],
            "needs_expert_review": dec_res["needs_expert_review"],
            "abstain_recommended": dec_res["abstain_recommended"],
            "decision_summary": dec_res["decision_summary"],
            "extracted_features": feats
        }
        pdf_path = ReportGeneratorService.generate_pdf_report(report_payload)
        db_record.report_pdf_path = f"/reports/{Path(pdf_path).name}"
        db.commit()

        return FullAnalysisResponse(
            analysis_id=db_record.id,
            patient_id=req.patient_id,
            modality=req.modality,
            scan_filename=db_record.scan_filename,
            created_at=db_record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            original_url=f"/uploads/{file_id}/original.png",
            denoised_url=prep_res["denoised_url"],
            enhanced_url=prep_res["enhanced_url"],
            normalized_url=prep_res["normalized_url"],
            segmentation_mask_url=seg_res["segmentation_mask_url"],
            tumor_overlay_url=seg_res["tumor_overlay_url"],
            extracted_region_url=seg_res["extracted_region_url"],
            tumor_detected=cls_res["tumor_detected"],
            predicted_class=dec_res["predicted_class"],
            confidence_score=cls_res["confidence_score"],
            class_probabilities=cls_res["class_probabilities"],
            extracted_features=feats,
            predictive_entropy=unc_res["predictive_entropy"],
            epistemic_uncertainty=unc_res["epistemic_uncertainty"],
            fuzzy_uncertainty_score=unc_res["fuzzy_uncertainty_score"],
            fuzzy_uncertainty_level=unc_res["fuzzy_uncertainty_level"],
            fuzzy_reasoning_explanation=unc_res["fuzzy_explanation"],
            fired_fuzzy_rules=unc_res["fired_fuzzy_rules"],
            needs_expert_review=dec_res["needs_expert_review"],
            abstain_recommended=dec_res["abstain_recommended"],
            decision_summary=dec_res["decision_summary"],
            report_pdf_url=db_record.report_pdf_path,
            model_status={
                "is_trained": cls_res["is_trained"],
                "model_name": cls_res["model_name"]
            }
        )
    except Exception as e:
        logger.error(f"Full analysis execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_analysis_history(
    modality: Optional[str] = None,
    tumor_status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc())
    if modality:
        query = query.filter(AnalysisHistory.modality == modality.upper())
    if tumor_status == "positive":
        query = query.filter(AnalysisHistory.tumor_detected == True)
    elif tumor_status == "negative":
        query = query.filter(AnalysisHistory.tumor_detected == False)
    elif tumor_status == "abstain":
        query = query.filter(AnalysisHistory.abstain_recommended == True)

    records = query.limit(limit).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "patient_id": r.patient_id,
            "modality": r.modality,
            "scan_filename": r.scan_filename,
            "original_url": r.original_filepath,
            "tumor_detected": r.tumor_detected,
            "predicted_class": r.predicted_class,
            "confidence_score": r.confidence_score,
            "fuzzy_uncertainty_level": r.fuzzy_uncertainty_level,
            "fuzzy_uncertainty_score": r.fuzzy_uncertainty_score,
            "needs_expert_review": r.needs_expert_review,
            "abstain_recommended": r.abstain_recommended,
            "report_pdf_url": r.report_pdf_path,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    return results

@router.get("/history/{analysis_id}")
def get_single_analysis(analysis_id: int, db: Session = Depends(get_db)):
    r = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    return {
        "id": r.id,
        "patient_id": r.patient_id,
        "modality": r.modality,
        "scan_filename": r.scan_filename,
        "original_url": r.original_filepath,
        "processed_filepaths": r.processed_filepaths,
        "segmentation_mask_url": r.segmentation_mask_path,
        "tumor_overlay_url": r.tumor_overlay_path,
        "extracted_region_url": r.extracted_region_path,
        "tumor_detected": r.tumor_detected,
        "predicted_class": r.predicted_class,
        "confidence_score": r.confidence_score,
        "class_probabilities": r.class_probabilities,
        "extracted_features": r.extracted_features,
        "predictive_entropy": r.predictive_entropy,
        "epistemic_uncertainty": r.epistemic_uncertainty,
        "fuzzy_uncertainty_level": r.fuzzy_uncertainty_level,
        "fuzzy_uncertainty_score": r.fuzzy_uncertainty_score,
        "fuzzy_reasoning_explanation": r.fuzzy_reasoning_explanation,
        "fired_fuzzy_rules": r.fired_fuzzy_rules,
        "needs_expert_review": r.needs_expert_review,
        "abstain_recommended": r.abstain_recommended,
        "decision_summary": r.decision_summary,
        "report_pdf_url": r.report_pdf_path,
        "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

@router.get("/report/{analysis_id}")
def download_pdf_report(analysis_id: int, db: Session = Depends(get_db)):
    r = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
    if not r or not r.report_pdf_path:
        raise HTTPException(status_code=404, detail="Report not found for analysis.")
    file_name = Path(r.report_pdf_path).name
    actual_path = settings.REPORTS_DIR / file_name
    if not actual_path.exists():
        raise HTTPException(status_code=404, detail="Report PDF file missing from disk.")
    return FileResponse(
        str(actual_path),
        media_type="application/pdf",
        filename=file_name
    )

@router.get("/model/status")
def get_model_status():
    _, is_mri_trained = TumorClassificationService.load_mri_model()
    _, is_ct_trained = TumorClassificationService.load_ct_model()
    _, is_seg_trained = TumorSegmentationService.load_model()
    return {
        "models": [
            {
                "name": "Brain MRI Classifier",
                "modality": "MRI",
                "architecture": "Custom Deep CNN with Monte Carlo Dropout (4 classes)",
                "checkpoint": str(settings.CHECKPOINTS_DIR / "cnn_mri.pt"),
                "is_trained": is_mri_trained,
                "classes": TumorClassificationService.MRI_CLASSES
            },
            {
                "name": "CT Space-Occupying Lesion Classifier",
                "modality": "CT",
                "architecture": "Custom Deep CNN with Monte Carlo Dropout (2 classes)",
                "checkpoint": str(settings.CHECKPOINTS_DIR / "cnn_ct.pt"),
                "is_trained": is_ct_trained,
                "classes": TumorClassificationService.CT_CLASSES
            },
            {
                "name": "Tumor Segmentation U-Net",
                "modality": "MRI",
                "architecture": "2D Convolutional U-Net Encoder-Decoder with Skip Connections",
                "checkpoint": str(settings.CHECKPOINTS_DIR / "unet_mri.pt"),
                "is_trained": is_seg_trained,
                "fallback_available": "Adaptive Intensity & Morphological Saliency Segmentation"
            }
        ],
        "active_device": "CPU"
    }

@router.get("/model/metrics")
def get_model_metrics():
    return ModelTrainingService.get_latest_metrics()

@router.post("/model/train")
def train_model(epochs: int = Query(5, ge=1, le=50), modality: str = Query("MRI")):
    try:
        metrics = ModelTrainingService.train_models(epochs=epochs, modality=modality)
        return {
            "status": "SUCCESS",
            "message": f"Successfully trained PyTorch CNN and U-Net models for {epochs} epochs.",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Training execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets")
def get_datasets_status():
    datasets = [
        {
            "id": "brats",
            "name": "BraTS (Brain Tumor Segmentation Challenge)",
            "modality": "MRI (T1, T1ce, T2, FLAIR)",
            "classes": ["Glioma (Necrotic/Active/Edema)", "No Tumor"],
            "status": "Research Framework Integrated",
            "description": "Standard benchmark dataset for glioma sub-region segmentation and brain tumor grading."
        },
        {
            "id": "figshare_brain",
            "name": "Figshare Brain Tumor MRI Dataset",
            "modality": "T1-weighted Contrast-Enhanced MRI",
            "classes": ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
            "status": "Label Schema Configured (3064 T1-CE axial, coronal, sagittal slices)",
            "description": "Established open medical imaging dataset with patient-level slice annotations."
        },
        {
            "id": "cq500_ct",
            "name": "Qure.ai CQ500 CT Dataset",
            "modality": "Head Non-Contrast CT",
            "classes": ["Intracranial Space-Occupying Lesion / Hemorrhage", "Normal"],
            "status": "Configured for CT Lesion Assessment",
            "description": "Curated head CT dataset for acute neurological lesions and midline shift detection."
        }
    ]
    return {"datasets": datasets}

@router.get("/knowledge-base")
def get_knowledge_base():
    kb = MedicalKnowledgeBaseService()
    return kb.kb_data

@router.post("/knowledge-base/rules")
def update_knowledge_base(payload: Dict[str, Any]):
    kb = MedicalKnowledgeBaseService()
    kb.save_knowledge_base(payload)
    return {"status": "SUCCESS", "message": "Knowledge base updated successfully."}

@router.get("/fuzzy/rules")
def get_fuzzy_rules():
    fuzzy = FuzzyReasoningEngine()
    return fuzzy.rules_config

@router.post("/fuzzy/rules")
def update_fuzzy_rules(payload: Dict[str, Any]):
    fuzzy = FuzzyReasoningEngine()
    fuzzy.save_config(payload)
    return {"status": "SUCCESS", "message": "Fuzzy reasoning rules and membership functions updated."}

@router.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "Medical Tumor Detection & Uncertainty Assessment API",
        "version": settings.VERSION
    }
