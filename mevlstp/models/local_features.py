"""
Handcrafted Local Feature Extraction.

Extracts sub-pixel boundary cues that coarse-resolution frozen encoders
smooth out: Gabor responses, Sobel edges, stroke-width estimates, CLAHE,
local variance, edge density, and gradient magnitude.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List
from PIL import Image


class TextLocalFeatures:
    """Extract handcrafted local features for polygon boundary refinement."""

    def __init__(self, size: int = 64):
        self.size = size
        # 12 Gabor kernels: 6 orientations × 2 frequencies
        self.gabor_kernels = []
        for theta in [0, 30, 60, 90, 120, 150]:
            for freq in [0.1, 0.2]:
                k = cv2.getGaborKernel(
                    (15, 15), 3.0, np.pi * theta / 180,
                    1.0 / freq, 0.5, 0, ktype=cv2.CV_32F,
                )
                self.gabor_kernels.append(k)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        # Auto-detect channel count
        dummy = np.zeros((size, size), dtype=np.float32)
        self.num_channels = self._extract_single(dummy).shape[0]

    def _extract_single(self, gray: np.ndarray) -> np.ndarray:
        """Extract features from a single grayscale image."""
        feats = []

        # Gabor responses (12 channels)
        for k in self.gabor_kernels:
            feats.append(np.abs(cv2.filter2D(gray, cv2.CV_32F, k)))

        # Sobel edges (2 channels)
        sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        feats.append(sx)
        feats.append(sy)

        # Stroke width transform approximation (2 channels)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        swt = np.ones_like(gray)
        m = edges > 0
        if m.any():
            swt[m] = 1.0 / (mag[m] + 1e-8)
        k3 = np.ones((3, 3), np.float32)
        feats.append(cv2.erode(swt.astype(np.float32), k3))
        feats.append(cv2.dilate(swt.astype(np.float32), k3))

        # CLAHE enhancement (1 channel)
        feats.append(self.clahe.apply((gray * 255).astype(np.uint8)) / 255.0)

        # Local variance (1 channel)
        feats.append(cv2.blur((gray - cv2.blur(gray, (5, 5))) ** 2, (5, 5)))

        # Edge density (1 channel)
        feats.append(cv2.blur(edges.astype(np.float32), (7, 7)) / 255.0)

        # Gradient magnitude (1 channel)
        feats.append(np.sqrt(sx ** 2 + sy ** 2))

        return np.stack(feats, axis=0).astype(np.float32)

    def extract_batch(
        self, images: List[Image.Image], device: str = "cuda"
    ) -> torch.Tensor:
        """
        Extract local features from a batch of PIL images.

        Args:
            images: List of PIL RGB images.
            device: Target device.

        Returns:
            Tensor of shape [B, C, size, size].
        """
        batch = []
        for img in images:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gray = cv2.resize(gray, (self.size, self.size))
            batch.append(torch.from_numpy(self._extract_single(gray)).float())
        return torch.stack(batch).to(device)


class LocalFeatureEncoder(nn.Module):
    """Convolutional encoder that processes raw local features to 512 channels."""

    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
