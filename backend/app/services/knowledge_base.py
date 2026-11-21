import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from app.core.config import settings
from app.core.logger import logger

class MedicalKnowledgeBaseService:
    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or (settings.KNOWLEDGE_BASE_DIR / "tumor_knowledge.json")
        self.kb_data = self.load_knowledge_base()

    def load_knowledge_base(self) -> Dict[str, Any]:
        if self.kb_path.exists():
            with open(self.kb_path, "r") as f:
                return json.load(f)
        return {"tumor_categories": []}

    def save_knowledge_base(self, new_data: Dict[str, Any]) -> None:
        self.kb_data = new_data
        with open(self.kb_path, "w") as f:
            json.dump(new_data, f, indent=2)

    def evaluate_features(
        self,
        features: Dict[str, Any],
        predicted_class: str,
        modality: str = "MRI"
    ) -> Dict[str, Any]:
        """
        Links extracted imaging & radiomic features to medical knowledge base profiles and rules.
        Calculates:
        1. Feature Consistency Score [0.0 - 1.0] with the predicted tumor class profile.
        2. Fired rules and clinical pattern explanations.
        3. Feature concordance summary for decision support.
        """
        intensity_feats = features.get("intensity_features", {})
        texture_feats = features.get("texture_features", {})
        shape_feats = features.get("shape_features", {})
        boundary_feats = features.get("boundary_characteristics", {})

        mean_val = intensity_feats.get("mean_intensity", 100.0)
        std_val = intensity_feats.get("std_intensity", 30.0)
        glcm_contrast = texture_feats.get("glcm_contrast", 20.0)
        glcm_homogeneity = texture_feats.get("glcm_homogeneity", 0.5)
        circularity = shape_feats.get("circularity", 0.5)
        boundary_irregularity = shape_feats.get("boundary_irregularity", 0.5)
        area_pixels = shape_feats.get("area_pixels", 0)

        # Context for rule evaluation
        context = {
            "mean_intensity": mean_val,
            "std_intensity": std_val,
            "glcm_contrast": glcm_contrast,
            "glcm_homogeneity": glcm_homogeneity,
            "circularity": circularity,
            "boundary_irregularity": boundary_irregularity,
            "area_pixels": area_pixels,
            "tumor_area": area_pixels,
            "max_contour_area": area_pixels
        }

        fired_rules = []
        best_category = None
        concordance_score = 0.50

        # Scan categories
        categories = self.kb_data.get("tumor_categories", [])
        for cat in categories:
            cat_id = cat.get("id", "").lower()
            if cat_id in predicted_class.lower() or predicted_class.lower() in cat_id:
                best_category = cat
                break

        # Check rules across all categories
        for cat in categories:
            for rule in cat.get("rules", []):
                condition = rule.get("condition", "")
                if self._eval_condition(condition, context):
                    fired_rules.append({
                        "category_name": cat.get("name"),
                        "rule_id": rule.get("rule_id"),
                        "condition": condition,
                        "inference": rule.get("inference"),
                        "confidence_weight": rule.get("confidence_weight", 0.8)
                    })

        # Calculate consistency with target category profile
        if best_category and "radiomic_feature_profiles" in best_category:
            profile = best_category["radiomic_feature_profiles"]
            matches = 0
            total_checks = 0
            for feat_name, bounds in profile.items():
                if feat_name in context:
                    val = context[feat_name]
                    min_b = bounds.get("min", -1e9)
                    max_b = bounds.get("max", 1e9)
                    total_checks += 1
                    if min_b <= val <= max_b:
                        matches += 1
            if total_checks > 0:
                concordance_score = matches / total_checks

        return {
            "predicted_class": predicted_class,
            "concordance_score": round(concordance_score, 3),
            "fired_rules": fired_rules,
            "matched_category": best_category.get("name") if best_category else "Unspecified Lesion",
            "imaging_characteristics": best_category.get("imaging_characteristics", {}) if best_category else {},
            "disclaimer": "These rules provide structured clinical correlation for decision support only and do not establish a definitive medical diagnosis."
        }

    @staticmethod
    def _eval_condition(cond_str: str, ctx: Dict[str, Any]) -> bool:
        """
        Safely evaluates simple boolean conditions like:
        'boundary_irregularity > 0.65 AND std_intensity > 40'
        """
        try:
            # Replace keywords
            expr = cond_str.replace("AND", "and").replace("OR", "or")
            # Safe evaluation with bounded namespace
            return bool(eval(expr, {"__builtins__": {}}, ctx))
        except Exception:
            return False
