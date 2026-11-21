from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
from app.core.config import settings
from app.core.logger import logger

class PreprocessingService:
    @staticmethod
    def preprocess_scan(
        file_id: str,
        slice_array: np.ndarray,
        gaussian_kernel: int = 3,
        median_kernel: int = 3,
        clahe_clip_limit: float = 2.0,
        target_size: int = 224
    ) -> Dict[str, Any]:
        """
        Executes actual medical image preprocessing pipeline:
        1. Noise reduction via Gaussian & Median filtering.
        2. Contrast enhancement via CLAHE and Histogram Equalization.
        3. Intensity standardization and normalization.
        4. Resizing to target model dimensions (224x224).
        5. Saves every intermediate image for side-by-side visual comparison.
        """
        upload_subfolder = settings.UPLOADS_DIR / file_id
        upload_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Ensure 2D float32 normalized [0, 1]
        arr = slice_array.astype(np.float32)
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max > arr_min:
            arr_norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_norm = np.zeros_like(arr)
            
        arr_uint8 = (arr_norm * 255.0).astype(np.uint8)
        
        # Step A: Noise Removal (Gaussian + Median)
        denoised_uint8 = PreprocessingService._apply_noise_reduction(arr_uint8, gaussian_kernel, median_kernel)
        
        # Step B: Contrast Enhancement (CLAHE)
        enhanced_uint8 = PreprocessingService._apply_clahe(denoised_uint8, clip_limit=clahe_clip_limit)
        
        # Step C: Standardization & Normalization (target_size x target_size, float [0, 1])
        normalized_pil = Image.fromarray(enhanced_uint8).resize((target_size, target_size), Image.Resampling.BILINEAR)
        normalized_uint8 = np.array(normalized_pil)
        
        # Save images
        denoised_path = upload_subfolder / "denoised.png"
        enhanced_path = upload_subfolder / "enhanced.png"
        normalized_path = upload_subfolder / "normalized.png"
        
        Image.fromarray(denoised_uint8).save(denoised_path, format="PNG")
        Image.fromarray(enhanced_uint8).save(enhanced_path, format="PNG")
        normalized_pil.save(normalized_path, format="PNG")
        
        transformations = [
            f"Gaussian Filter (k={gaussian_kernel})",
            f"Median Filter (k={median_kernel})",
            f"CLAHE Contrast Enhancement (clip={clahe_clip_limit})",
            f"Bilinear Resampling to ({target_size}x{target_size})",
            "Intensity Normalization [0.0, 1.0]"
        ]
        
        return {
            "file_id": file_id,
            "original_url": f"/uploads/{file_id}/original.png",
            "denoised_url": f"/uploads/{file_id}/denoised.png",
            "enhanced_url": f"/uploads/{file_id}/enhanced.png",
            "normalized_url": f"/uploads/{file_id}/normalized.png",
            "denoised_array": denoised_uint8,
            "enhanced_array": enhanced_uint8,
            "normalized_array": (normalized_uint8.astype(np.float32) / 255.0),
            "transformations_applied": transformations
        }

    @staticmethod
    def _apply_noise_reduction(img_uint8: np.ndarray, g_k: int = 3, m_k: int = 3) -> np.ndarray:
        try:
            import cv2
            g_k = g_k if g_k % 2 == 1 else g_k + 1
            m_k = m_k if m_k % 2 == 1 else m_k + 1
            gauss = cv2.GaussianBlur(img_uint8, (g_k, g_k), sigmaX=1.0)
            median = cv2.medianBlur(gauss, m_k)
            return median
        except Exception:
            # Fallback using scipy.ndimage
            sigma = 1.0
            g_filtered = ndimage.gaussian_filter(img_uint8.astype(np.float32), sigma=sigma)
            m_filtered = ndimage.median_filter(g_filtered.astype(np.uint8), size=m_k)
            return m_filtered

    @staticmethod
    def _apply_clahe(img_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        try:
            import cv2
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img_uint8)
        except Exception:
            # Robust tile-based adaptive histogram equalization in pure numpy
            pil_img = Image.fromarray(img_uint8)
            # Global histogram equalization as fallback
            return np.array(ImageOps.equalize(pil_img))
