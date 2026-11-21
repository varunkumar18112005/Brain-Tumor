from typing import Dict, Any, Tuple
import numpy as np
from scipy import ndimage
from app.core.logger import logger

class FeatureExtractionService:
    @staticmethod
    def extract_all_features(
        scan_slice: np.ndarray,
        tumor_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive numerical features from segmented medical lesion:
        1. First-Order Intensity Statistics (Mean, Std, Skewness, Min, Max, Range).
        2. GLCM Texture Descriptors (Contrast, Dissimilarity, Homogeneity, Energy, ASM, Correlation, Entropy).
        3. Morphological & Shape Descriptors (Area, Perimeter, Circularity, Boundary Irregularity, Solidity, Aspect Ratio).
        4. Spatial & Dimension Characteristics (Bounding dimensions, Equivalent Diameter).
        """
        arr = scan_slice.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
            
        has_lesion = bool(np.sum(tumor_mask) > 10)
        
        if has_lesion:
            lesion_pixels = arr[tumor_mask > 0]
        else:
            # Fallback to general brain parenchyma if no lesion detected
            brain_mask = arr > 20
            lesion_pixels = arr[brain_mask] if np.sum(brain_mask) > 0 else arr.flatten()

        # 1. Intensity Statistics
        mean_val = float(np.mean(lesion_pixels))
        std_val = float(np.std(lesion_pixels))
        min_val = float(np.min(lesion_pixels))
        max_val = float(np.max(lesion_pixels))
        range_val = float(max_val - min_val)
        
        # Skewness: m3 / (m2^(3/2))
        diff = lesion_pixels - mean_val
        m2 = np.mean(diff ** 2)
        m3 = np.mean(diff ** 3)
        skewness = float(m3 / (m2 ** 1.5 + 1e-8))

        # 2. GLCM Texture Descriptors (Computed on quantized 32-level ROI)
        glcm_feats = FeatureExtractionService._compute_glcm_features(arr, tumor_mask if has_lesion else None)

        # 3. Shape Descriptors
        shape_feats = FeatureExtractionService._compute_shape_features(tumor_mask)

        # 4. Dimension Characteristics
        equiv_diameter = float(2.0 * np.sqrt(shape_feats["area_pixels"] / np.pi))

        return {
            "intensity_features": {
                "mean_intensity": round(mean_val, 2),
                "std_intensity": round(std_val, 2),
                "min_intensity": round(min_val, 2),
                "max_intensity": round(max_val, 2),
                "intensity_range": round(range_val, 2),
                "skewness": round(skewness, 3)
            },
            "texture_features": {
                "glcm_contrast": round(glcm_feats["contrast"], 2),
                "glcm_dissimilarity": round(glcm_feats["dissimilarity"], 2),
                "glcm_homogeneity": round(glcm_feats["homogeneity"], 3),
                "glcm_energy": round(glcm_feats["energy"], 4),
                "glcm_asm": round(glcm_feats["asm"], 4),
                "glcm_correlation": round(glcm_feats["correlation"], 3),
                "glcm_entropy": round(glcm_feats["entropy"], 3)
            },
            "shape_features": {
                "area_pixels": shape_feats["area_pixels"],
                "perimeter_pixels": round(shape_feats["perimeter"], 2),
                "circularity": round(shape_feats["circularity"], 3),
                "boundary_irregularity": round(shape_feats["boundary_irregularity"], 3),
                "aspect_ratio": round(shape_feats["aspect_ratio"], 2),
                "solidity": round(shape_feats["solidity"], 3),
                "equivalent_diameter": round(equiv_diameter, 2)
            },
            "boundary_characteristics": {
                "clarity_score": round(1.0 - shape_feats["boundary_irregularity"], 3),
                "margin_description": "Smooth and circumscribed" if shape_feats["boundary_irregularity"] < 0.45 else "Irregular and infiltrative",
                "lesion_detected": has_lesion
            }
        }

    @staticmethod
    def _compute_glcm_features(img_255: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculates Gray-Level Co-occurrence Matrix (GLCM) at distance=1 across 4 angles (0, 45, 90, 135 degrees).
        """
        # Quantize to 32 gray levels for fast, robust texture matrix
        n_levels = 32
        quantized = np.clip((img_255 / 256.0 * n_levels).astype(int), 0, n_levels - 1)
        
        # Directions: (dr, dc)
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        h, w = quantized.shape
        
        total_glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
        
        for dr, dc in directions:
            # Shifted arrays
            r_start, r_end = max(0, -dr), min(h, h - dr)
            c_start, c_end = max(0, -dc), min(w, w - dc)
            
            orig_patch = quantized[r_start:r_end, c_start:c_end]
            neighbor_patch = quantized[r_start + dr:r_end + dr, c_start + dc:c_end + dc]
            
            if mask is not None:
                mask_patch = mask[r_start:r_end, c_start:c_end] > 0
                orig_patch = orig_patch[mask_patch]
                neighbor_patch = neighbor_patch[mask_patch]
            else:
                orig_patch = orig_patch.flatten()
                neighbor_patch = neighbor_patch.flatten()
                
            if len(orig_patch) > 0:
                for i, j in zip(orig_patch, neighbor_patch):
                    total_glcm[i, j] += 1
                    total_glcm[j, i] += 1  # Symmetrical GLCM

        # Normalize probabilities
        glcm_sum = np.sum(total_glcm)
        if glcm_sum > 0:
            P = total_glcm / glcm_sum
        else:
            P = np.eye(n_levels) / n_levels

        # Feature formulas
        i_idx, j_idx = np.indices((n_levels, n_levels))
        
        contrast = float(np.sum(P * ((i_idx - j_idx) ** 2)))
        dissimilarity = float(np.sum(P * np.abs(i_idx - j_idx)))
        homogeneity = float(np.sum(P / (1.0 + (i_idx - j_idx) ** 2)))
        asm = float(np.sum(P ** 2))
        energy = float(np.sqrt(asm))
        
        # Entropy
        p_nonzero = P[P > 0]
        entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        
        # Correlation
        mu_i = np.sum(i_idx * P)
        mu_j = np.sum(j_idx * P)
        var_i = np.sum(((i_idx - mu_i) ** 2) * P)
        var_j = np.sum(((j_idx - mu_j) ** 2) * P)
        cov = np.sum((i_idx - mu_i) * (j_idx - mu_j) * P)
        correlation = float(cov / (np.sqrt(var_i * var_j) + 1e-8))
        correlation = min(1.0, max(-1.0, correlation))

        return {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "asm": asm,
            "energy": energy,
            "correlation": correlation,
            "entropy": entropy
        }

    @staticmethod
    def _compute_shape_features(mask: np.ndarray) -> Dict[str, Any]:
        coords = np.argwhere(mask > 0)
        area = int(len(coords))
        if area == 0:
            return {
                "area_pixels": 0, "perimeter": 0.0, "circularity": 0.0,
                "boundary_irregularity": 0.0, "aspect_ratio": 1.0, "solidity": 0.0
            }

        # Perimeter via boundary erosion
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        perimeter = float(np.sum(mask ^ eroded))

        if perimeter > 0:
            circularity = (4.0 * np.pi * float(area)) / (perimeter ** 2)
            circularity = min(1.0, max(0.0, circularity))
        else:
            circularity = 0.0

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        h_box = r_max - r_min + 1
        w_box = c_max - c_min + 1
        aspect_ratio = float(w_box / h_box) if h_box > 0 else 1.0

        # Solidity = Area / Convex Hull Area (approximated via filled bounding box)
        bbox_area = float(h_box * w_box)
        solidity = float(area / bbox_area) if bbox_area > 0 else 1.0

        return {
            "area_pixels": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "boundary_irregularity": 1.0 - circularity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity
        }
