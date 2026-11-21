export interface UploadResponse {
  file_id: string;
  original_filename: string;
  filepath: string;
  preview_path: string;
  relative_url: string;
  dimensions: number[];
  format: string;
  filesize_kb: number;
  modality: 'MRI' | 'CT';
  patient_id: string;
  total_slices: number;
  selected_slice: number;
}

export interface PreprocessingResponse {
  file_id: string;
  original_url: string;
  denoised_url: string;
  enhanced_url: string;
  normalized_url: string;
  transformations_applied: string[];
}

export interface SegmentationResponse {
  file_id: string;
  segmentation_mask_url: string;
  tumor_overlay_url: string;
  extracted_region_url: string;
  tumor_area_pixels: number;
  tumor_perimeter_pixels: number;
  circularity: number;
  boundary_irregularity: number;
  bounding_box: {
    min_row: number;
    max_row: number;
    min_col: number;
    max_col: number;
    width: number;
    height: number;
  };
  model_used: string;
  is_trained_model: boolean;
  dice_score?: number | null;
  iou_score?: number | null;
}

export interface ExtractedFeatures {
  intensity_features: {
    mean_intensity: number;
    std_intensity: number;
    min_intensity: number;
    max_intensity: number;
    intensity_range: number;
    skewness: number;
  };
  texture_features: {
    glcm_contrast: number;
    glcm_dissimilarity: number;
    glcm_homogeneity: number;
    glcm_energy: number;
    glcm_asm: number;
    glcm_correlation: number;
    glcm_entropy: number;
  };
  shape_features: {
    area_pixels: number;
    perimeter_pixels: number;
    circularity: number;
    boundary_irregularity: number;
    aspect_ratio: number;
    solidity: number;
    equivalent_diameter: number;
  };
  boundary_characteristics: {
    clarity_score: number;
    margin_description: string;
    lesion_detected: boolean;
  };
}

export interface FiredRule {
  category_name?: string;
  rule_id: string;
  firing_strength?: number;
  consequent?: string;
  explanation?: string;
  inference?: string;
  conditions?: string[];
}

export interface FullAnalysisResult {
  analysis_id: number;
  patient_id: string;
  modality: string;
  scan_filename: string;
  created_at: string;
  original_url: string;
  denoised_url: string;
  enhanced_url: string;
  normalized_url: string;
  segmentation_mask_url: string;
  tumor_overlay_url: string;
  extracted_region_url: string;
  tumor_detected: boolean;
  predicted_class: string;
  confidence_score: number;
  class_probabilities: Record<string, number>;
  extracted_features: ExtractedFeatures;
  predictive_entropy: number;
  epistemic_uncertainty: number;
  fuzzy_uncertainty_score: number;
  fuzzy_uncertainty_level: string;
  fuzzy_reasoning_explanation: string;
  fired_fuzzy_rules: FiredRule[];
  needs_expert_review: boolean;
  abstain_recommended: boolean;
  decision_summary: string;
  report_pdf_url?: string;
  model_status: {
    is_trained: boolean;
    model_name: string;
  };
}

export interface AnalysisHistoryItem {
  id: number;
  patient_id: string;
  modality: string;
  scan_filename: string;
  original_url: string;
  tumor_detected: boolean;
  predicted_class: string;
  confidence_score: number;
  fuzzy_uncertainty_level: string;
  fuzzy_uncertainty_score: number;
  needs_expert_review: boolean;
  abstain_recommended: boolean;
  report_pdf_url: string;
  created_at: string;
}

export interface ModelMetrics {
  status?: string;
  accuracy: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
  class_names: string[];
  dice_coefficient?: number;
  iou_score?: number;
  ece: number;
  reliability_diagram: {
    bin_range: string;
    confidence: number;
    accuracy: number;
    samples: number;
  }[];
  entropy_distribution: {
    range: string;
    count: number;
  }[];
  training_history: {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  };
  last_trained: string;
}

export interface ModelStatus {
  models: {
    name: string;
    modality: string;
    architecture: string;
    checkpoint: string;
    is_trained: boolean;
    classes?: string[];
    fallback_available?: string;
  }[];
  active_device: string;
}
