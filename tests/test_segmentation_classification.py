import pytest
import numpy as np
import torch
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from ml.models.unet import UNet
from ml.models.cnn_classifier import MedicalTumorClassifier
from app.services.segmentation import TumorSegmentationService
from app.services.classification import TumorClassificationService

def test_unet_architecture():
    model = UNet(in_channels=1, out_channels=1, base_filters=16)
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)
    assert 0.0 <= out.min().item()
    assert out.max().item() <= 1.0

def test_cnn_classifier_mc_dropout():
    classifier = MedicalTumorClassifier(in_channels=1, num_classes=4, dropout_rate=0.3)
    x = torch.randn(1, 1, 224, 224)
    mc_samples = classifier.forward_mc(x, samples=10)
    
    # Shape: [samples, 1, num_classes]
    assert mc_samples.shape == (10, 1, 4)
    # Check that probabilities sum to 1.0
    for s in range(10):
        sum_p = mc_samples[s, 0].sum().item()
        assert abs(sum_p - 1.0) < 1e-4

def test_dice_and_iou_calculation():
    mask_a = np.zeros((50, 50), dtype=np.uint8)
    mask_b = np.zeros((50, 50), dtype=np.uint8)
    mask_a[10:30, 10:30] = 1
    mask_b[10:30, 10:30] = 1
    
    dice, iou = TumorSegmentationService.calculate_dice_iou(mask_a, mask_b)
    assert dice == pytest.approx(1.0, rel=1e-4)
    assert iou == pytest.approx(1.0, rel=1e-4)
