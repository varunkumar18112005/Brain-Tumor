import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net architecture for medical tumor segmentation.
    Accepts 1-channel or 3-channel medical scan and produces a 1-channel binary segmentation mask.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, base_filters, dropout_rate=0.0)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters, base_filters * 2, dropout_rate=dropout_rate)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters * 2, base_filters * 4, dropout_rate=dropout_rate)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters * 4, base_filters * 8, dropout_rate=dropout_rate)
        )
        
        # Bottleneck
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters * 8, base_filters * 16, dropout_rate=dropout_rate)
        )

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)

        # Output Projection
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)
