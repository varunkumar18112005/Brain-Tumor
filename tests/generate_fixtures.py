"""
Synthetic Test Scan Generator
Explicitly for automated software testing and verification.
NOT clinical scans.
"""
from pathlib import Path
import numpy as np
from PIL import Image

def generate_test_fixtures():
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    img_size = 256
    y, x = np.ogrid[:img_size, :img_size]
    cy, cx = img_size // 2, img_size // 2

    # 1. MRI Glioma-like scan
    brain_mask = ((x - cx) ** 2) / (0.42 * img_size) ** 2 + ((y - cy) ** 2) / (0.46 * img_size) ** 2 <= 1.0
    mri_arr = np.zeros((img_size, img_size), dtype=np.float32)
    mri_arr[brain_mask] = np.random.normal(0.40, 0.05, size=np.sum(brain_mask))
    
    # Infiltrative tumor mass
    ty, tx = cy - 20, cx + 25
    tumor_mask = ((x - tx) ** 2 + (y - ty) ** 2 <= 28 ** 2) & brain_mask
    mri_arr[tumor_mask] = np.random.normal(0.85, 0.12, size=np.sum(tumor_mask))
    mri_arr = np.clip(mri_arr, 0.0, 1.0)
    Image.fromarray((mri_arr * 255.0).astype(np.uint8)).save(fixtures_dir / "sample_mri_tumor.png")

    # 2. MRI Normal Brain scan
    norm_arr = np.zeros((img_size, img_size), dtype=np.float32)
    norm_arr[brain_mask] = np.random.normal(0.42, 0.04, size=np.sum(brain_mask))
    # Ventricles
    v_mask = (((x - cx + 14) ** 2) / 6 ** 2 + ((y - cy) ** 2) / 24 ** 2 <= 1.0) | \
             (((x - cx - 14) ** 2) / 6 ** 2 + ((y - cy) ** 2) / 24 ** 2 <= 1.0)
    norm_arr[v_mask & brain_mask] = 0.15
    norm_arr = np.clip(norm_arr, 0.0, 1.0)
    Image.fromarray((norm_arr * 255.0).astype(np.uint8)).save(fixtures_dir / "sample_mri_normal.png")

    # 3. CT Lesion scan
    ct_arr = np.zeros((img_size, img_size), dtype=np.float32)
    head_mask = ((x - cx) ** 2 + (y - cy) ** 2 <= (0.44 * img_size) ** 2)
    skull_ring = head_mask & ~((x - cx) ** 2 + (y - cy) ** 2 <= (0.40 * img_size) ** 2)
    parenchyma = ((x - cx) ** 2 + (y - cy) ** 2 <= (0.40 * img_size) ** 2)
    ct_arr[skull_ring] = 0.98  # Dense bone
    ct_arr[parenchyma] = np.random.normal(0.35, 0.04, size=np.sum(parenchyma))
    # Focal hyperdense hematoma/lesion
    ct_arr[((x - cx - 25) ** 2 + (y - cy + 15) ** 2 <= 18 ** 2) & parenchyma] = 0.78
    ct_arr = np.clip(ct_arr, 0.0, 1.0)
    Image.fromarray((ct_arr * 255.0).astype(np.uint8)).save(fixtures_dir / "sample_ct_lesion.png")

if __name__ == "__main__":
    generate_test_fixtures()
    print("Synthetic test fixtures generated in tests/fixtures/")
