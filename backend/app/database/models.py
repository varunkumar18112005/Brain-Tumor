from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, JSON
from app.database.connection import Base

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String(64), index=True, nullable=False)
    modality = Column(String(16), nullable=False)  # MRI or CT
    scan_filename = Column(String(255), nullable=False)
    original_filepath = Column(String(512), nullable=False)
    
    # Processed Image Filepaths (Stored as JSON dictionary of URLs or relative paths)
    processed_filepaths = Column(JSON, nullable=True)
    segmentation_mask_path = Column(String(512), nullable=True)
    tumor_overlay_path = Column(String(512), nullable=True)
    extracted_region_path = Column(String(512), nullable=True)
    
    # Classification Findings
    tumor_detected = Column(Boolean, nullable=True)
    predicted_class = Column(String(64), nullable=True)
    confidence_score = Column(Float, nullable=True)
    class_probabilities = Column(JSON, nullable=True)  # Dict of class -> prob
    
    # Radiomic & Extracted Features
    extracted_features = Column(JSON, nullable=True)
    
    # Uncertainty Assessment
    predictive_entropy = Column(Float, nullable=True)
    epistemic_uncertainty = Column(Float, nullable=True)
    aleatoric_uncertainty = Column(Float, nullable=True)
    fuzzy_uncertainty_score = Column(Float, nullable=True)
    fuzzy_uncertainty_level = Column(String(32), nullable=True)  # Low, Moderate, High
    fuzzy_reasoning_explanation = Column(Text, nullable=True)
    fired_fuzzy_rules = Column(JSON, nullable=True)
    
    # Decision Support
    needs_expert_review = Column(Boolean, default=False)
    abstain_recommended = Column(Boolean, default=False)
    decision_summary = Column(Text, nullable=True)
    
    # PDF Report Path
    report_pdf_path = Column(String(512), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(128), unique=True, index=True)
    task = Column(String(64))  # classification or segmentation
    modality = Column(String(16))  # MRI or CT
    architecture = Column(String(128))
    checkpoint_path = Column(String(512), nullable=True)
    is_trained = Column(Boolean, default=False)
    metrics = Column(JSON, nullable=True)
    last_trained = Column(DateTime, default=datetime.utcnow)
