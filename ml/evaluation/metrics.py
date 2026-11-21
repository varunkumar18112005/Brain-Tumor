from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
)

class MedicalEvaluationMetrics:
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates genuine medical classification performance metrics:
        Accuracy, Sensitivity, Specificity, Precision, Recall, F1-score, and Confusion Matrix.
        """
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        
        # Multiclass confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        
        # Per-class Specificity calculation: TN / (TN + FP)
        specificities = []
        for i in range(len(class_names)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = float(tn / (tn + fp + 1e-8))
            specificities.append(spec)
        mean_specificity = float(np.mean(specificities))

        # Expected Calibration Error (ECE) and Reliability Diagram Data
        ece, reliability_bins = MedicalEvaluationMetrics.calculate_calibration(y_true, y_probs, num_bins=10)

        # Predictive Entropy Distribution
        entropy_hist = MedicalEvaluationMetrics.calculate_entropy_distribution(y_probs)

        return {
            "accuracy": round(acc, 4),
            "sensitivity": round(float(recall), 4),
            "recall": round(float(recall), 4),
            "specificity": round(mean_specificity, 4),
            "precision": round(float(precision), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "ece": round(ece, 4),
            "reliability_diagram": reliability_bins,
            "entropy_distribution": entropy_hist
        }

    @staticmethod
    def calculate_calibration(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_bins: int = 10
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Calculates Expected Calibration Error (ECE) and Reliability diagram bin points.
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n_total = len(confidences)
        bins_data = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if np.sum(in_bin) > 0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confidences[in_bin]))
                count_in_bin = int(np.sum(in_bin))
                ece += (count_in_bin / n_total) * abs(acc_in_bin - conf_in_bin)
            else:
                acc_in_bin = 0.0
                conf_in_bin = float((bin_lower + bin_upper) / 2.0)
                count_in_bin = 0
                
            bins_data.append({
                "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "confidence": round(conf_in_bin, 3),
                "accuracy": round(acc_in_bin, 3),
                "samples": count_in_bin
            })

        return float(ece), bins_data

    @staticmethod
    def calculate_entropy_distribution(y_probs: np.ndarray) -> List[Dict[str, Any]]:
        # Compute Shannon entropy per prediction
        probs = np.clip(y_probs, 1e-12, 1.0)
        entropies = -np.sum(probs * np.log2(probs), axis=1)
        max_ent = np.log2(y_probs.shape[1])
        norm_ent = entropies / max_ent if max_ent > 0 else entropies
        
        hist, edges = np.histogram(norm_ent, bins=6, range=(0.0, 1.0))
        dist = []
        for count, e_low, e_high in zip(hist, edges[:-1], edges[1:]):
            dist.append({
                "range": f"{e_low:.2f}-{e_high:.2f}",
                "count": int(count)
            })
        return dist
