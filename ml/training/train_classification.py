#!/usr/bin/env python3
"""
Medical CNN Tumor Classification & MC Dropout Training Script
Usage:
    python ml/training/train_classification.py --epochs 5 --modality MRI
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR))

from app.services.model_trainer import ModelTrainingService

def main():
    parser = argparse.ArgumentParser(description="Train Medical Tumor Classification & Segmentation Models")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--modality", type=str, default="MRI", choices=["MRI", "CT"], help="Target modality")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print(f"===========================================================")
    print(f" Starting Training: Modality={args.modality}, Epochs={args.epochs}")
    print(f"===========================================================")
    
    metrics = ModelTrainingService.train_models(
        epochs=args.epochs,
        modality=args.modality,
        batch_size=args.batch_size
    )

    print("\nTraining completed successfully!")
    print(f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Sensitivity (Recall): {metrics['sensitivity'] * 100:.2f}%")
    print(f"Specificity:         {metrics['specificity'] * 100:.2f}%")
    print(f"Precision:           {metrics['precision'] * 100:.2f}%")
    print(f"F1-Score:            {metrics['f1_score'] * 100:.2f}%")
    print(f"Dice Coefficient:    {metrics.get('dice_coefficient', 'N/A')}")
    print(f"IoU (Jaccard):       {metrics.get('iou_score', 'N/A')}")
    print(f"Expected Calib Err:  {metrics['ece']:.4f}")
    print(f"Checkpoints saved to ml/checkpoints/")

if __name__ == "__main__":
    main()
