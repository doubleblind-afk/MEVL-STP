"""
PSE Decoder with Deep Supervision.

Progressive Scale Expansion decoder that predicts K kernel maps at
increasing scales. Includes auxiliary prediction heads at intermediate
resolutions for deep supervision during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn import CBAM, SEBlock


class MultiScaleDecoder(nn.Module):
    """
    Decoder with deep supervision.

    Upsamples the fused FPN features to full resolution and produces
    the main prediction plus two auxiliary predictions at 1/2 and 1/4
    resolution for deep supervision.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(), CBAM(256),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), SEBlock(128),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
        )

        # Main prediction head (full resolution)
        self.head_main = nn.Conv2d(64, 1, 1)
        # Auxiliary heads for deep supervision
        self.head_aux2 = nn.Conv2d(128, 1, 1)  # 1/2 resolution
        self.head_aux1 = nn.Conv2d(256, 1, 1)  # 1/4 resolution

    def forward(self, x):
        """
        Args:
            x: Fused feature map [B, C, H, W] from FPN + local features.

        Returns:
            Tuple of (main, aux_half, aux_quarter) prediction maps.
        """
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        aux1 = self.head_aux1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        aux2 = self.head_aux2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = self.up4(x)
        main = self.head_main(x)

        return main, aux2, aux1
