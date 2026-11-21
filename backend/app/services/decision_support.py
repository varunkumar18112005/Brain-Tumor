from typing import Dict, Any, List
from datetime import datetime

class DecisionSupportService:
    @staticmethod
    def generate_clinical_summary(
        patient_id: str,
        modality: str,
        classification: Dict[str, Any],
        uncertainty: Dict[str, Any],
        features: Dict[str, Any],
        is_trained_model: bool
    ) -> Dict[str, Any]:
        """
        Synthesizes deep model outputs, radiomic feature findings, knowledge base rules,
        and fuzzy uncertainty metrics into a coherent, responsible decision support statement.
        """
        pred_class = classification.get("predicted_class", "Unspecified")
        conf = classification.get("confidence_score", 0.0)
        entropy = uncertainty.get("predictive_entropy", 0.0)
        fuzzy_level = uncertainty.get("fuzzy_uncertainty_level", "Moderate Uncertainty")
        fuzzy_score = uncertainty.get("fuzzy_uncertainty_score", 0.5)
        needs_review = uncertainty.get("needs_expert_review", False)
        abstain = uncertainty.get("abstain_recommended", False)
        tumor_detected = classification.get("tumor_detected", False)

        intensity_f = features.get("intensity_features", {})
        shape_f = features.get("shape_features", {})
        texture_f = features.get("texture_features", {})

        summary_paragraphs = []

        # 1. Primary Finding
        if abstain:
            summary_paragraphs.append(
                f"[ABSTAIN / INCONCLUSIVE]: Model confidence ({conf*100:.1f}%) and predictive entropy "
                f"({entropy:.2f}) exceed safety reliability bounds. The system formally refrains from issuing an automated "
                f"classification. Mandatory human radiological evaluation required."
            )
        elif not tumor_detected:
            summary_paragraphs.append(
                f"Primary Finding: No focal space-occupying lesion detected. Scan characteristics align with normal parenchymal "
                f"imaging patterns with {conf*100:.1f}% model confidence ({fuzzy_level})."
            )
        else:
            summary_paragraphs.append(
                f"Primary Finding: Suspicious focal lesion identified. Most probable classification: {pred_class} "
                f"(Model Confidence: {conf*100:.1f}%, Predictive Entropy: {entropy:.2f}, Fuzzy Assessment: {fuzzy_level})."
            )

        # 2. Key Radiomic Feature Correlates
        area = shape_f.get("area_pixels", 0)
        circularity = shape_f.get("circularity", 0.0)
        irregularity = shape_f.get("boundary_irregularity", 0.0)
        mean_int = intensity_f.get("mean_intensity", 0.0)
        contrast = texture_f.get("glcm_contrast", 0.0)

        if tumor_detected and area > 0:
            margin_desc = "irregular, infiltrative margins" if irregularity > 0.50 else "smooth, circumscribed margins"
            summary_paragraphs.append(
                f"Morphological Profile: Segmented lesion encompasses approximately {area} pixels with {margin_desc} "
                f"(circularity index: {circularity:.2f}, boundary irregularity: {irregularity:.2f}). "
                f"Internal texture exhibits mean intensity of {mean_int:.1f} with GLCM contrast of {contrast:.1f}."
            )

        # 3. Model Information & Training Status
        if not is_trained_model:
            summary_paragraphs.append(
                "[NOTICE]: Deep neural network weights have not yet completed dataset training. Findings are derived from "
                "baseline radiomic Bayesian priors. Train the model using the Model Performance suite to activate deep inference."
            )

        # 4. Expert Review & Safety Disclaimer
        if needs_review:
            summary_paragraphs.append(
                "WARNING: High or intermediate diagnostic uncertainty flagged by the Fuzzy Reasoning Engine. "
                "Corroborating clinical laboratory findings and specialized multi-sequence MRI/contrast CT review are strongly recommended."
            )

        summary_paragraphs.append(
            "DISCLAIMER: This system is an academic research demonstration and intelligent decision support tool. "
            "It does NOT provide definitive clinical diagnoses and must NOT replace licensed medical specialists or radiologists."
        )

        full_text = "\n\n".join(summary_paragraphs)

        return {
            "patient_id": patient_id,
            "modality": modality,
            "tumor_detected": tumor_detected,
            "predicted_class": "Inconclusive / Abstain" if abstain else pred_class,
            "confidence_score": conf,
            "predictive_entropy": entropy,
            "fuzzy_uncertainty_level": fuzzy_level,
            "fuzzy_uncertainty_score": fuzzy_score,
            "needs_expert_review": needs_review,
            "abstain_recommended": abstain,
            "decision_summary": full_text
        }
