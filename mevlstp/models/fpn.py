"""
Hierarchical Feature Pyramid Network with Channel Attention.

Fuses multi-resolution encoder features via top-down lateral connections
with CBAM and SE attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], -1, 1, 1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 4)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels, r), nn.ReLU(), nn.Linear(r, channels), nn.Sigmoid()
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_c = x.mean(dim=[2, 3])
        max_c = x.amax(dim=[2, 3])
        ch_attn = (self.ch_fc(avg_c) + self.ch_fc(max_c)).view(x.shape[0], -1, 1, 1)
        x = x * ch_attn
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.amax(dim=1, keepdim=True)
        sp_attn = self.sp_conv(torch.cat([avg_s, max_s], dim=1))
        return x * sp_attn


class LateralFPN(nn.Module):
    """
    Hierarchical FPN with lateral skip connections.

    Takes features at multiple resolutions from the encoder ensemble,
    projects each to a common embedding dimension, then fuses them
    top-down (coarse to fine) with CBAM refinement at each level.
    """

    def __init__(self, in_dims: List[int], embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # 1×1 lateral projections
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            for d in in_dims
        ])

        # Top-down refinement blocks
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                CBAM(embed_dim),
            )
            for _ in range(len(in_dims) - 1)
        ])

        # Final output with SE attention
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            SEBlock(embed_dim),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, C_i, H_i, W_i] tensors from encoders.

        Returns:
            Fused feature map [B, embed_dim, H_finest, W_finest].
        """
        # Project each feature to embed_dim (preserving original order)
        projected = [self.lateral[i](features[i]) for i in range(len(features))]

        # Sort by spatial resolution (coarse → fine) for top-down fusion
        order = sorted(
            range(len(projected)),
            key=lambda i: projected[i].shape[2] * projected[i].shape[3],
        )

        fused = projected[order[0]]
        for j, idx in enumerate(order[1:]):
            h, w = projected[idx].shape[2], projected[idx].shape[3]
            fused = F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
            fused = fused + projected[idx]
            fused = self.refine[j](fused)

        return self.output_conv(fused)
