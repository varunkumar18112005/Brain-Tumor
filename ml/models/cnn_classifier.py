import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class MedicalTumorClassifier(nn.Module):
    """
    Deep CNN with explicit Dropout layers enabled during inference for Monte Carlo Dropout
    predictive uncertainty estimation.
    Default classes: Glioma, Meningioma, Pituitary, No Tumor (4 classes for Brain MRI).
    Or Space-Occupying Lesion vs Normal (2 classes for CT).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, dropout_rate: float = 0.35):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 224 -> 112
            ConvBlock(32, 64),            # 112 -> 56
            ConvBlock(64, 128),           # 56 -> 28
            ConvBlock(128, 256),          # 28 -> 14
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def forward_mc(self, x: torch.Tensor, samples: int = 15) -> torch.Tensor:
        """
        Executes Monte Carlo Dropout forward passes with dropout forced active
        while keeping batch normalization in eval mode to support single-sample (batch_size=1) inference.
        Returns tensor of shape [samples, batch_size, num_classes] with softmax probabilities.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
        prob_samples = []
        with torch.no_grad():
            for _ in range(samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                prob_samples.append(probs)
        return torch.stack(prob_samples, dim=0)
