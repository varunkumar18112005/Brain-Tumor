import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

from app.core.config import settings
from app.core.logger import logger
from ml.models.unet import UNet
from ml.models.cnn_classifier import MedicalTumorClassifier
from ml.evaluation.metrics import MedicalEvaluationMetrics
from app.services.classification import TumorClassificationService
from app.services.segmentation import TumorSegmentationService

class ModelTrainingService:
    @staticmethod
    def generate_synthetic_research_dataset(num_samples: int = 160, img_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates realistic synthetic brain MRI slice patterns and masks for research validation:
        Class 0: Glioma (Infiltrative, heterogeneous lesion with irregular margin)
        Class 1: Meningioma (Circumscribed, hyperintense extra-axial/dural mass)
        Class 2: Pituitary (Focal central sellar base mass)
        Class 3: No Tumor (Normal brain parenchyma with ventricles)
        """
        np.random.seed(42)
        images = []
        masks = []
        labels = []

        for i in range(num_samples):
            cls_idx = i % 4
            img = np.zeros((img_size, img_size), dtype=np.float32)
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            
            y, x = np.ogrid[:img_size, :img_size]
            cy, cx = img_size // 2, img_size // 2
            
            # Realistic cranium / skull and brain parenchyma
            skull = ((x - cx) ** 2) / (0.45 * img_size) ** 2 + ((y - cy) ** 2) / (0.48 * img_size) ** 2 <= 1.0
            brain = ((x - cx) ** 2) / (0.42 * img_size) ** 2 + ((y - cy) ** 2) / (0.45 * img_size) ** 2 <= 1.0
            
            img[skull] = 0.22
            img[brain] = np.random.normal(0.40, 0.05, size=np.sum(brain))
            
            # Bilateral central CSF ventricles
            v_mask = (((x - cx + 16) ** 2) / 8 ** 2 + ((y - cy) ** 2) / 32 ** 2 <= 1.0) | \
                     (((x - cx - 16) ** 2) / 8 ** 2 + ((y - cy) ** 2) / 32 ** 2 <= 1.0)
            img[v_mask & brain] = 0.12

            if cls_idx == 0:  # Glioma
                ty = cy + np.random.randint(-25, 25)
                tx = cx + np.random.randint(-30, 30)
                r = np.random.randint(24, 40)
                theta = np.arctan2(y - ty, x - tx)
                r_var = r + 8.0 * np.sin(3.0 * theta) + 4.0 * np.cos(5.0 * theta)
                t_mask = ((x - tx) ** 2 + (y - ty) ** 2 <= r_var ** 2) & brain
                mask[t_mask] = 1.0
                img[t_mask] = np.clip(np.random.normal(0.78, 0.14, size=np.sum(t_mask)), 0.5, 1.0)
                labels.append(0)
            elif cls_idx == 1:  # Meningioma
                ty = cy - 48
                tx = cx + 40
                r = np.random.randint(18, 30)
                t_mask = ((x - tx) ** 2 + (y - ty) ** 2 <= r ** 2) & brain
                mask[t_mask] = 1.0
                img[t_mask] = np.clip(0.88 + np.random.normal(0, 0.03, size=np.sum(t_mask)), 0.7, 1.0)
                labels.append(1)
            elif cls_idx == 2:  # Pituitary
                ty = cy + 42
                tx = cx
                r = np.random.randint(14, 22)
                t_mask = ((x - tx) ** 2 + (y - ty) ** 2 <= r ** 2) & brain
                mask[t_mask] = 1.0
                img[t_mask] = np.clip(0.78 + np.random.normal(0, 0.04, size=np.sum(t_mask)), 0.65, 0.95)
                labels.append(2)
            else:  # No Tumor
                labels.append(3)

            img = np.clip(img, 0.0, 1.0)
            images.append(img)
            masks.append(mask)

        img_tensor = torch.tensor(np.array(images, dtype=np.float32)).unsqueeze(1)
        mask_tensor = torch.tensor(np.array(masks, dtype=np.float32)).unsqueeze(1)
        label_tensor = torch.tensor(np.array(labels, dtype=np.int64))
        return img_tensor, mask_tensor, label_tensor

    @staticmethod
    def train_models(
        epochs: int = 5,
        modality: str = "MRI",
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Executes actual training on PyTorch models:
        1. Trains MedicalTumorClassifier CNN with CrossEntropyLoss & Dropout.
        2. Trains UNet with BCEWithLogits / BCELoss for segmentation.
        3. Computes real training & validation curves across epochs.
        4. Calculates accuracy, precision, recall, F1, specificity, ECE, Dice, IoU.
        5. Saves checkpoints in ml/checkpoints/ and updates live system services.
        """
        logger.info(f"Starting model training pipeline for modality: {modality} (Epochs: {epochs})")
        
        img_t, mask_t, label_t = ModelTrainingService.generate_synthetic_research_dataset(num_samples=160, img_size=224)
        
        # Train / Validation Split (80 / 20)
        n_samples = len(img_t)
        n_train = int(n_samples * 0.8)
        
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        train_loader = DataLoader(
            TensorDataset(img_t[train_idx], mask_t[train_idx], label_t[train_idx]),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(img_t[val_idx], mask_t[val_idx], label_t[val_idx]),
            batch_size=batch_size, shuffle=False
        )

        # 1. Train Classifier
        classifier = MedicalTumorClassifier(in_channels=1, num_classes=4, dropout_rate=0.25)
        cls_optimizer = optim.Adam(classifier.parameters(), lr=5e-4)
        cls_criterion = nn.CrossEntropyLoss()

        # 2. Train UNet Segmentation
        unet = UNet(in_channels=1, out_channels=1, base_filters=32)
        unet_optimizer = optim.Adam(unet.parameters(), lr=1e-3)
        unet_criterion = nn.BCELoss()

        history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            classifier.train()
            unet.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for bx, bm, by in train_loader:
                # Forward classifier
                cls_optimizer.zero_grad()
                logits = classifier(bx)
                loss_c = cls_criterion(logits, by)
                loss_c.backward()
                cls_optimizer.step()

                # Forward UNet
                unet_optimizer.zero_grad()
                pred_m = unet(bx)
                loss_u = unet_criterion(pred_m, bm)
                loss_u.backward()
                unet_optimizer.step()

                epoch_loss += loss_c.item() * len(by)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()
                total += len(by)

            train_loss = epoch_loss / total
            train_acc = correct / total

            # Validation
            classifier.eval()
            unet.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bx, bm, by in val_loader:
                    logits = classifier(bx)
                    loss_c = cls_criterion(logits, by)
                    val_loss += loss_c.item() * len(by)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == by).sum().item()
                    val_total += len(by)

            v_loss = val_loss / val_total
            v_acc = val_correct / val_total

            history["epochs"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(v_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(v_acc, 4))
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Acc: {v_acc:.4f}")

        # Compute Final Comprehensive Evaluation on Validation Set
        val_imgs = img_t[val_idx]
        val_masks = mask_t[val_idx]
        val_labels = label_t[val_idx]

        with torch.no_grad():
            classifier.eval()
            logits = classifier(val_imgs)
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)
            y_true = val_labels.numpy()

            # Segmentation validation
            pred_masks = unet(val_imgs).numpy()
            bin_preds = (pred_masks > 0.5).astype(np.uint8)
            bin_true = (val_masks.numpy() > 0.5).astype(np.uint8)
            
            dices = []
            ious = []
            for bp, bt in zip(bin_preds, bin_true):
                d, iou = TumorSegmentationService.calculate_dice_iou(bp[0], bt[0])
                dices.append(d)
                ious.append(iou)
            mean_dice = float(np.mean(dices))
            mean_iou = float(np.mean(ious))

        class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        eval_metrics = MedicalEvaluationMetrics.calculate_classification_metrics(
            y_true, y_pred, probs, class_names
        )
        eval_metrics["dice_coefficient"] = round(mean_dice, 4)
        eval_metrics["iou_score"] = round(mean_iou, 4)
        eval_metrics["training_history"] = history
        eval_metrics["last_trained"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Save Checkpoints
        TumorClassificationService.set_trained_weights("MRI", classifier.state_dict())
        TumorSegmentationService.set_weights(unet.state_dict())

        # Save Metrics Record
        metrics_file = settings.CHECKPOINTS_DIR / "mri_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        return eval_metrics

    @staticmethod
    def get_latest_metrics() -> Dict[str, Any]:
        metrics_file = settings.CHECKPOINTS_DIR / "mri_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                return json.load(f)
        return {
            "status": "NOT_TRAINED",
            "message": "Model weights have not been trained yet. Trigger training from this dashboard or run train_classification.py."
        }
