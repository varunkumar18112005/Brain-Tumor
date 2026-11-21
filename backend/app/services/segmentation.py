from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from app.core.config import settings
from app.core.logger import logger
from ml.models.unet import UNet

class TumorSegmentationService:
    _model: Optional[UNet] = None
    _is_trained: bool = False
    _checkpoint_path: Path = settings.CHECKPOINTS_DIR / "unet_mri.pt"

    @classmethod
    def load_model(cls) -> Tuple[UNet, bool]:
        if cls._model is None:
            cls._model = UNet(in_channels=1, out_channels=1, base_filters=32)
            if cls._checkpoint_path.exists():
                try:
                    state_dict = torch.load(str(cls._checkpoint_path), map_location="cpu", weights_only=True)
                    cls._model.load_state_dict(state_dict)
                    cls._model.eval()
                    cls._is_trained = True
                    logger.info("Loaded trained UNet segmentation weights.")
                except Exception as e:
                    logger.warning(f"Failed to load UNet weights: {e}")
                    cls._is_trained = False
            else:
                cls._is_trained = False
                logger.info("UNet weights checkpoint not found. Operating in baseline segmentation mode.")
        return cls._model, cls._is_trained

    @classmethod
    def set_weights(cls, state_dict: dict):
        cls._model = UNet(in_channels=1, out_channels=1, base_filters=32)
        cls._model.load_state_dict(state_dict)
        cls._model.eval()
        cls._is_trained = True
        torch.save(state_dict, str(cls._checkpoint_path))

    @staticmethod
    def segment(
        file_id: str,
        norm_slice: np.ndarray,
        modality: str = "MRI",
        use_deep_model: bool = True
    ) -> Dict[str, Any]:
        """
        Executes tumor segmentation:
        1. Ensures standard normalized input array in [0.0, 1.0].
        2. Evaluates PyTorch U-Net with skip connections.
        3. Corroborates with adaptive intensity and morphological contouring.
        4. Calculates area, perimeter, circularity, boundary irregularity.
        5. Generates high-contrast overlay and bounding box ROI crop.
        """
        upload_subfolder = settings.UPLOADS_DIR / file_id
        upload_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Ensure canonical [0.0, 1.0] scaling
        arr = norm_slice.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        
        h, w = arr.shape[:2]
        model, is_trained = TumorSegmentationService.load_model()
        
        mask = np.zeros((h, w), dtype=np.uint8)
        model_used = "Classical Adaptive Intensity & Gradient Thresholding"
        
        if use_deep_model and is_trained:
            try:
                # Prepare tensor [1, 1, 224, 224] without uint8 overflow
                img_pil = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((224, 224))
                tensor_in = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    pred_mask = model(tensor_in).squeeze().cpu().numpy()
                bin_mask_224 = (pred_mask > 0.5).astype(np.uint8)
                mask = np.array(Image.fromarray(bin_mask_224).resize((w, h), Image.Resampling.NEAREST))
                model_used = "Deep U-Net (PyTorch 2D Convolutional)"
            except Exception as e:
                logger.error(f"Deep U-Net inference error: {e}")
                mask = TumorSegmentationService._classical_adaptive_segmentation(arr)
        else:
            mask = TumorSegmentationService._classical_adaptive_segmentation(arr)

        # Corroborate: if deep U-Net found minimal pixels on a real scan with clear focal lesion
        if np.sum(mask) < 80:
            adaptive_mask = TumorSegmentationService._classical_adaptive_segmentation(arr)
            if np.sum(adaptive_mask) >= 80:
                mask = adaptive_mask
                model_used = "Corroborated Adaptive Morphological Segmentation"

        # Morphological post-processing: keep largest connected component if present
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = (labeled == largest_label).astype(np.uint8)
            
        area_pixels = int(np.sum(mask))
        perimeter, circularity, irregularity, bbox = TumorSegmentationService._calculate_contour_metrics(mask)
        
        # Save Visualizations
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_path = upload_subfolder / "segmentation_mask.png"
        Image.fromarray(mask_uint8).save(mask_path, format="PNG")
        
        # Color Overlay (Red highlight over grayscale original)
        base_gray = (arr * 255.0).astype(np.uint8)
        color_overlay = np.stack([base_gray, base_gray, base_gray], axis=-1)
        red_mask = mask.astype(bool)
        color_overlay[red_mask, 0] = np.clip(color_overlay[red_mask, 0] * 0.4 + 220, 0, 255).astype(np.uint8)
        color_overlay[red_mask, 1] = np.clip(color_overlay[red_mask, 1] * 0.4, 0, 255).astype(np.uint8)
        color_overlay[red_mask, 2] = np.clip(color_overlay[red_mask, 2] * 0.4, 0, 255).astype(np.uint8)
        
        # Draw contour boundary in bright cyan
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        boundary = mask ^ eroded
        color_overlay[boundary, 0] = 0
        color_overlay[boundary, 1] = 255
        color_overlay[boundary, 2] = 255
        
        overlay_path = upload_subfolder / "tumor_overlay.png"
        Image.fromarray(color_overlay).save(overlay_path, format="PNG")
        
        # Extracted Region
        extracted = np.zeros_like(color_overlay)
        extracted[red_mask] = color_overlay[red_mask]
        if bbox["width"] > 0 and bbox["height"] > 0:
            pad = 10
            r0 = max(0, bbox["min_row"] - pad)
            r1 = min(h, bbox["max_row"] + pad)
            c0 = max(0, bbox["min_col"] - pad)
            c1 = min(w, bbox["max_col"] + pad)
            cropped_roi = extracted[r0:r1, c0:c1]
        else:
            cropped_roi = extracted
            
        roi_path = upload_subfolder / "extracted_region.png"
        Image.fromarray(cropped_roi).save(roi_path, format="PNG")
        
        return {
            "file_id": file_id,
            "segmentation_mask_url": f"/uploads/{file_id}/segmentation_mask.png",
            "tumor_overlay_url": f"/uploads/{file_id}/tumor_overlay.png",
            "extracted_region_url": f"/uploads/{file_id}/extracted_region.png",
            "tumor_area_pixels": area_pixels,
            "tumor_perimeter_pixels": round(float(perimeter), 2),
            "circularity": round(float(circularity), 3),
            "boundary_irregularity": round(float(irregularity), 3),
            "bounding_box": bbox,
            "model_used": model_used,
            "is_trained_model": is_trained,
            "dice_score": None,
            "iou_score": None,
            "mask_array": mask
        }

    @staticmethod
    def _classical_adaptive_segmentation(norm_slice: np.ndarray) -> np.ndarray:
        """
        Robust Otsu + foreground brain tissue masking + morphological saliency thresholding.
        Identifies focal hyperintense / hypointense lesion region on actual patient scan.
        """
        arr = norm_slice.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
            
        # 1. Mask background (air outside skull)
        brain_mask = arr > 0.08
        if np.sum(brain_mask) == 0:
            return np.zeros_like(arr, dtype=np.uint8)
            
        brain_pixels = arr[brain_mask]
        p75 = np.percentile(brain_pixels, 75)
        p90 = np.percentile(brain_pixels, 90)
        mean_val = np.mean(brain_pixels)
        std_val = np.std(brain_pixels)
        
        # High intensity focal candidate
        high_candidate = (arr >= max(p90, mean_val + 1.1 * std_val)) & brain_mask
        
        # Morphological opening to eliminate point noise, then closing to unify solid lesion
        opened = ndimage.binary_opening(high_candidate, structure=np.ones((3, 3)))
        closed = ndimage.binary_closing(opened, structure=np.ones((5, 5)))
        
        # Ensure minimum lesion size
        labeled, num_features = ndimage.label(closed)
        if num_features == 0:
            return np.zeros_like(arr, dtype=np.uint8)
            
        sizes = ndimage.sum(closed, labeled, range(1, num_features + 1))
        # Find largest component with at least 50 pixels
        valid_indices = np.where(sizes >= 50)[0]
        if len(valid_indices) == 0:
            return np.zeros_like(arr, dtype=np.uint8)
            
        best_idx = valid_indices[np.argmax(sizes[valid_indices])] + 1
        return (labeled == best_idx).astype(np.uint8)

    @staticmethod
    def _calculate_contour_metrics(mask: np.ndarray) -> Tuple[float, float, float, Dict[str, int]]:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return 0.0, 0.0, 0.0, {"min_row": 0, "max_row": 0, "min_col": 0, "max_col": 0, "width": 0, "height": 0}
            
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        bbox = {
            "min_row": int(r_min), "max_row": int(r_max),
            "min_col": int(c_min), "max_col": int(c_max),
            "width": int(c_max - c_min + 1), "height": int(r_max - r_min + 1)
        }
        
        area = float(len(coords))
        # Boundary perimeter estimation
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        boundary_pixels = mask ^ eroded
        perimeter = float(np.sum(boundary_pixels))
        
        # Circularity: 4 * pi * Area / (Perimeter^2)
        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter ** 2)
            circularity = min(1.0, max(0.0, circularity))
        else:
            circularity = 0.0
            
        # Irregularity index: 1.0 - circularity
        irregularity = 1.0 - circularity
        
        return perimeter, circularity, irregularity, bbox

    @staticmethod
    def calculate_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
        intersection = np.sum((pred_mask > 0) & (gt_mask > 0))
        pred_sum = np.sum(pred_mask > 0)
        gt_sum = np.sum(gt_mask > 0)
        union = np.sum((pred_mask > 0) | (gt_mask > 0))
        
        dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
        iou = intersection / (union + 1e-8)
        return float(dice), float(iou)
