#!/usr/bin/env python3
"""
Medical U-Net Tumor Segmentation Training Script
Usage:
    python ml/training/train_segmentation.py --epochs 5
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.services.model_trainer import ModelTrainingService

def main():
    parser = argparse.ArgumentParser(description="Train Medical U-Net Segmentation Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    print(f"Training U-Net Segmentation Model for {args.epochs} epochs...")
    metrics = ModelTrainingService.train_models(epochs=args.epochs, modality="MRI")
    print(f"Done! Dice: {metrics.get('dice_coefficient')}, IoU: {metrics.get('iou_score')}")

if __name__ == "__main__":
    main()
