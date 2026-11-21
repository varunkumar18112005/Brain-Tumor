from typing import Dict, Any, List, Tuple
import numpy as np
from app.core.config import settings
from app.core.logger import logger
from app.services.fuzzy_engine import FuzzyReasoningEngine
from app.services.knowledge_base import MedicalKnowledgeBaseService

class UncertaintyAssessmentService:
    def __init__(self):
        self.fuzzy_engine = FuzzyReasoningEngine()
        self.kb_service = MedicalKnowledgeBaseService()

    def assess(
        self,
        classification_result: Dict[str, Any],
        segmentation_result: Dict[str, Any],
        extracted_features: Dict[str, Any],
        modality: str = "MRI"
    ) -> Dict[str, Any]:
        """
        Synthesizes Dual Uncertainty Assessment:
        Part A: Probabilistic Reasoning (Monte Carlo Dropout, Shannon Predictive Entropy, Epistemic vs Aleatoric).
        Part B: Fuzzy Reasoning (Mamdani Inference Engine linking confidence, segmentation, boundary clarity, & radiomic consistency).
        Part C: Final Decision-Support synthesis & Abstention Thresholding.
        """
        confidence_score = float(classification_result.get("confidence_score", 0.5))
        class_probabilities = classification_result.get("class_probabilities", {})
        mc_sample_matrix = np.array(classification_result.get("mc_sample_matrix", []))  # [samples, num_classes]

        # 1. Probabilistic Uncertainty
        probs_array = np.array(list(class_probabilities.values()), dtype=np.float64)
        probs_array = np.clip(probs_array, 1e-12, 1.0)
        probs_array = probs_array / np.sum(probs_array)  # Normalize
        num_classes = len(probs_array)

        # Shannon Predictive Entropy: H(p) = - sum(p * log2(p))
        predictive_entropy = float(-np.sum(probs_array * np.log2(probs_array)))
        max_entropy = float(np.log2(num_classes)) if num_classes > 1 else 1.0
        normalized_entropy = float(predictive_entropy / max_entropy) if max_entropy > 0 else 0.0

        # Epistemic vs Aleatoric Uncertainty via MC Dropout Decomposition
        if len(mc_sample_matrix) > 1:
            # Expected entropy across passes (Aleatoric proxy: data noise)
            sample_entropies = [-np.sum(p * np.log2(np.clip(p, 1e-12, 1.0))) for p in mc_sample_matrix]
            mean_sample_entropy = float(np.mean(sample_entropies))
            aleatoric_uncertainty = float(mean_sample_entropy / max_entropy) if max_entropy > 0 else 0.0
            
            # Mutual Information (Epistemic proxy: model weight uncertainty)
            # I(y, W | x) = H(mean_p) - E[H(p_t)]
            epistemic_uncertainty = float(max(0.0, normalized_entropy - aleatoric_uncertainty))
            
            # Variance across MC passes
            prediction_variability = float(np.mean(np.var(mc_sample_matrix, axis=0)))
        else:
            aleatoric_uncertainty = float(normalized_entropy * 0.4)
            epistemic_uncertainty = float(normalized_entropy * 0.6)
            prediction_variability = float((1.0 - confidence_score) * 0.1)

        # 2. Extract inputs for Fuzzy Engine
        # A. Segmentation Quality: based on solidity and valid area
        shape_feats = extracted_features.get("shape_features", {})
        boundary_feats = extracted_features.get("boundary_characteristics", {})
        area = shape_feats.get("area_pixels", 0)
        circularity = shape_feats.get("circularity", 0.5)
        boundary_clarity = float(boundary_feats.get("clarity_score", 0.5))

        if area > 100:
            solidity = float(shape_feats.get("solidity", 0.7))
            segmentation_quality = float(np.clip(solidity * 0.7 + (1.0 - shape_feats.get("boundary_irregularity", 0.5)) * 0.3, 0.0, 1.0))
        else:
            # No tumor detected -> high segmentation quality for absence
            segmentation_quality = 0.90
            boundary_clarity = 0.85

        # B. Feature Consistency with Medical Knowledge Base
        predicted_class = classification_result.get("predicted_class", "No Tumor")
        kb_eval = self.kb_service.evaluate_features(extracted_features, predicted_class, modality)
        feature_consistency = float(kb_eval.get("concordance_score", 0.5))

        # 3. Fuzzy Inference Evaluation
        fuzzy_eval = self.fuzzy_engine.assess_uncertainty(
            prediction_confidence=confidence_score,
            segmentation_quality=segmentation_quality,
            boundary_clarity=boundary_clarity,
            feature_consistency=feature_consistency
        )

        fuzzy_uncertainty_score = fuzzy_eval["fuzzy_uncertainty_score"]
        fuzzy_uncertainty_level = fuzzy_eval["fuzzy_uncertainty_level"]
        fired_rules = fuzzy_eval["fired_rules"]
        fuzzy_explanation = fuzzy_eval["fuzzy_reasoning_explanation"]

        # 4. Final Combined Decision Support Flagging
        # Low confidence (< 0.55) OR High Predictive Entropy (> 0.65) OR High Fuzzy Uncertainty -> Flag
        needs_expert_review = (
            fuzzy_eval["needs_expert_review"] or
            (confidence_score < settings.EXPERT_REVIEW_THRESHOLD) or
            (normalized_entropy > 0.60) or
            (fuzzy_uncertainty_score > 0.55)
        )
        
        abstain_recommended = (
            fuzzy_eval["abstain_recommended"] or
            (confidence_score < settings.ABSTENTION_THRESHOLD and normalized_entropy > 0.70) or
            (fuzzy_uncertainty_score >= 0.75)
        )

        return {
            "confidence_score": round(confidence_score, 4),
            "predictive_entropy": round(normalized_entropy, 4),
            "raw_entropy_bits": round(predictive_entropy, 3),
            "epistemic_uncertainty": round(epistemic_uncertainty, 4),
            "aleatoric_uncertainty": round(aleatoric_uncertainty, 4),
            "prediction_variability": round(prediction_variability, 5),
            "fuzzy_uncertainty_score": round(fuzzy_uncertainty_score, 3),
            "fuzzy_uncertainty_level": fuzzy_uncertainty_level,
            "fuzzy_explanation": fuzzy_explanation,
            "fired_fuzzy_rules": fired_rules,
            "fuzzified_inputs": fuzzy_eval["fuzzified_inputs"],
            "needs_expert_review": needs_expert_review,
            "abstain_recommended": abstain_recommended,
            "knowledge_base_concordance": kb_eval
        }
