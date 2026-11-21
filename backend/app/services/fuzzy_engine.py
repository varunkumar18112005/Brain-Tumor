import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from app.core.config import settings
from app.core.logger import logger

class FuzzyReasoningEngine:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (settings.KNOWLEDGE_BASE_DIR / "fuzzy_rules.json")
        self.rules_config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def save_config(self, new_config: Dict[str, Any]) -> None:
        self.rules_config = new_config
        with open(self.config_path, "w") as f:
            json.dump(new_config, f, indent=2)

    @staticmethod
    def membership_value(x: float, params: List[float]) -> float:
        """
        Calculates membership degree for triangular (3 points) or trapezoidal (4 points) fuzzy sets.
        """
        x = float(np.clip(x, 0.0, 1.0))
        if len(params) == 3:
            # Triangular: a, b, c
            a, b, c = params
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b > a else 1.0
            else:
                return (c - x) / (c - b) if c > b else 1.0
        elif len(params) == 4:
            # Trapezoidal: a, b, c, d
            a, b, c, d = params
            if x <= a or x >= d:
                return 0.0
            elif a < x < b:
                return (x - a) / (b - a) if b > a else 1.0
            elif b <= x <= c:
                return 1.0
            else:
                return (d - x) / (d - c) if d > c else 1.0
        return 0.0

    def fuzzify_inputs(
        self,
        prediction_confidence: float,
        segmentation_quality: float,
        boundary_clarity: float,
        feature_consistency: float
    ) -> Dict[str, Dict[str, float]]:
        mfs = self.rules_config.get("membership_functions", {})
        inputs = {
            "prediction_confidence": prediction_confidence,
            "segmentation_quality": segmentation_quality,
            "boundary_clarity": boundary_clarity,
            "feature_consistency": feature_consistency
        }
        fuzzified = {}
        for var_name, val in inputs.items():
            fuzzified[var_name] = {}
            var_mfs = mfs.get(var_name, {}).get("sets", {})
            for set_name, params in var_mfs.items():
                fuzzified[var_name][set_name] = self.membership_value(val, params)
        return fuzzified

    def evaluate_rules(
        self,
        fuzzified: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Evaluates Mamdani fuzzy rules using Min (AND) operator.
        Returns consequent activation levels and list of fired rules with explanations.
        """
        rules = self.rules_config.get("rules", [])
        output_activations = {"low": 0.0, "moderate": 0.0, "high": 0.0}
        fired_rules = []

        for rule in rules:
            antecedents = rule.get("if", {})
            consequent = rule.get("then", {}).get("uncertainty_level", "moderate")
            
            # T-Norm (Min) across all conditions
            firing_strength = 1.0
            condition_evals = []
            for var_name, target_set in antecedents.items():
                degree = fuzzified.get(var_name, {}).get(target_set, 0.0)
                condition_evals.append(f"{var_name} is {target_set} (μ={degree:.2f})")
                firing_strength = min(firing_strength, degree)

            if firing_strength > 0.05:  # Significant rule activation
                output_activations[consequent] = max(output_activations.get(consequent, 0.0), firing_strength)
                fired_rules.append({
                    "rule_id": rule.get("id"),
                    "firing_strength": round(firing_strength, 3),
                    "consequent": consequent,
                    "explanation": rule.get("explanation"),
                    "conditions": condition_evals
                })

        return output_activations, fired_rules

    def defuzzify(self, activations: Dict[str, float]) -> float:
        """
        Centroid Defuzzification over universe [0, 1] with 101 sample points.
        """
        output_mfs = self.rules_config.get("membership_functions", {}).get("uncertainty_level", {}).get("sets", {})
        z_points = np.linspace(0.0, 1.0, 101)
        aggregated_mf = np.zeros_like(z_points)

        for set_name, weight in activations.items():
            if weight > 0 and set_name in output_mfs:
                params = output_mfs[set_name]
                for idx, z in enumerate(z_points):
                    deg = self.membership_value(z, params)
                    clipped_deg = min(deg, weight)  # Mamdani clipping
                    aggregated_mf[idx] = max(aggregated_mf[idx], clipped_deg)

        sum_mf = np.sum(aggregated_mf)
        if sum_mf == 0:
            # Default fallback based on weighted average of centers
            centers = {"low": 0.15, "moderate": 0.50, "high": 0.85}
            tot_w = sum(activations.values())
            if tot_w > 0:
                return float(sum(centers[k] * activations[k] for k in centers) / tot_w)
            return 0.50

        centroid = np.sum(z_points * aggregated_mf) / sum_mf
        return float(np.clip(centroid, 0.0, 1.0))

    def assess_uncertainty(
        self,
        prediction_confidence: float,
        segmentation_quality: float,
        boundary_clarity: float,
        feature_consistency: float
    ) -> Dict[str, Any]:
        """
        Full fuzzy reasoning pipeline:
        Fuzzification -> Rule Evaluation -> Defuzzification -> Natural Language Explanation.
        """
        fuzzified = self.fuzzify_inputs(
            prediction_confidence, segmentation_quality, boundary_clarity, feature_consistency
        )
        activations, fired_rules = self.evaluate_rules(fuzzified)
        uncertainty_score = self.defuzzify(activations)

        if uncertainty_score < 0.35:
            level = "Low Uncertainty"
            needs_review = False
        elif uncertainty_score < 0.65:
            level = "Moderate Uncertainty"
            needs_review = (prediction_confidence < 0.70)
        else:
            level = "High Uncertainty"
            needs_review = True

        thresholds = self.rules_config.get("decision_thresholds", {})
        abstain_threshold = thresholds.get("abstention_threshold", 0.55)
        abstain = (prediction_confidence < abstain_threshold) or (uncertainty_score >= 0.75)

        # Build natural-language explanation
        explanation_lines = [
            f"Fuzzy Rule-Based Uncertainty Assessment: Evaluated score {uncertainty_score:.2f} ({level}).",
            f"Input Parameters: Model Confidence={prediction_confidence*100:.1f}%, Segmentation Quality={segmentation_quality*100:.1f}%, "
            f"Boundary Clarity={boundary_clarity*100:.1f}%, Feature Consistency={feature_consistency*100:.1f}."
        ]
        if fired_rules:
            explanation_lines.append(f"{len(fired_rules)} rule(s) activated:")
            for r in fired_rules:
                explanation_lines.append(f"• [{r['rule_id']} - Strength {r['firing_strength']:.2f}]: {r['explanation']}")
        else:
            explanation_lines.append("No specialized edge rules activated; baseline moderate certainty applied.")

        if abstain:
            explanation_lines.append("RECOMMENDATION: Model abstains from automated diagnosis due to high uncertainty. Mandatory expert radiologist review required.")
        elif needs_review:
            explanation_lines.append("RECOMMENDATION: Intermediate uncertainty detected; clinical correlation and secondary expert review recommended.")
        else:
            explanation_lines.append("RECOMMENDATION: Low uncertainty observed across neural and radiomic indicators. AI decision support findings concordant.")

        return {
            "fuzzy_uncertainty_score": round(uncertainty_score, 3),
            "fuzzy_uncertainty_level": level,
            "fired_rules": fired_rules,
            "fuzzified_inputs": fuzzified,
            "activations": {k: round(v, 3) for k, v in activations.items()},
            "needs_expert_review": needs_review,
            "abstain_recommended": abstain,
            "fuzzy_reasoning_explanation": "\n".join(explanation_lines),
            "is_rule_based_assessment": True
        }
