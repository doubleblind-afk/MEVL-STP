"""
Multi-Encoder Feature Extraction Module.

Six frozen vision encoders spanning three complementary feature spectra:
  - Semantic: CLIP, SigLIP, EVA-CLIP (language-image pretraining)
  - Spatial:  DINOv2, SAM (self-supervised / segmentation)
  - Texture:  ConvNeXt (hierarchical CNN)

All encoders are kept frozen. We extract the last N hidden states from each
transformer encoder and average them to produce richer feature maps.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
from PIL import Image

from transformers import (
    CLIPImageProcessor, CLIPVisionModel,
    AutoImageProcessor, Dinov2Model,
    SiglipImageProcessor, SiglipVisionModel,
    AutoModel,
)
import timm


class MultiEncoderExtractor:
    """Wrapper for the six frozen foundation encoders."""

    def __init__(self, device="cuda", num_layers=4, dtype=torch.float16):
        self.device = device
        self.num_layers = num_layers
        self.dtype = dtype

        # CLIP ViT-L/14-336 (Semantic)
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", torch_dtype=dtype
        ).to(device).eval()

        # DINOv2-Large (Spatial)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large"
        )
        self.dino_model = Dinov2Model.from_pretrained(
            "facebook/dinov2-large", torch_dtype=dtype
        ).to(device).eval()

        # SigLIP-SO400M (Semantic)
        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384", torch_dtype=dtype
        ).to(device).eval()

        # EVA-CLIP ViT-L/14 (Semantic)
        self.eva_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.eva_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=dtype
        ).to(device).eval()

        # SAM ViT-H (Spatial)
        self.sam_processor = AutoImageProcessor.from_pretrained(
            "facebook/sam-vit-huge"
        )
        self.sam_model = AutoModel.from_pretrained(
            "facebook/sam-vit-huge", torch_dtype=dtype
        ).vision_encoder.to(device).eval()

        # ConvNeXt-Large (Texture) — stages 1, 2, 3
        self.convnext_model = timm.create_model(
            "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
            pretrained=True, features_only=True, out_indices=[1, 2, 3],
        ).to(device).eval().to(dtype)

        # Freeze all parameters
        for m in [
            self.clip_model, self.dino_model, self.siglip_model,
            self.eva_model, self.sam_model, self.convnext_model,
        ]:
            for p in m.parameters():
                p.requires_grad = False

    def _process_transformer(self, processor, model, images, has_cls=True):
        """Extract multi-layer averaged features from a transformer encoder."""
        B = len(images)
        inp = processor(images=images, return_tensors="pt")
        inp = {k: v.to(self.device).to(self.dtype) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        layers = torch.stack(out.hidden_states[-self.num_layers:], dim=0)
        avg = layers.mean(0)
        if has_cls:
            avg = avg[:, 1:, :]  # Remove CLS token
        avg = avg.float()
        gs = int(np.sqrt(avg.shape[1]))
        feat = avg[:, :gs * gs, :].reshape(B, gs, gs, -1).permute(0, 3, 1, 2)
        return feat

    @torch.no_grad()
    def extract(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Extract features from all six encoders.

        Args:
            images: List of PIL images (batch).

        Returns:
            List of feature tensors [B, C, H, W] at various resolutions.
        """
        B = len(images)
        all_feats = []

        # CLIP (24×24 patches @ 336px)
        all_feats.append(
            self._process_transformer(
                self.clip_processor, self.clip_model, images, has_cls=True
            )
        )

        # DINOv2
        all_feats.append(
            self._process_transformer(
                self.dino_processor, self.dino_model, images, has_cls=True
            )
        )

        # SigLIP (no CLS token)
        all_feats.append(
            self._process_transformer(
                self.siglip_processor, self.siglip_model, images, has_cls=False
            )
        )

        # EVA-CLIP
        all_feats.append(
            self._process_transformer(
                self.eva_processor, self.eva_model, images, has_cls=True
            )
        )

        # SAM
        inp = self.sam_processor(images=images, return_tensors="pt")
        inp = {k: v.to(self.device).to(self.dtype) for k, v in inp.items()}
        with torch.no_grad():
            out = self.sam_model(inp["pixel_values"])
        sam_feat = (out["last_hidden_state"] if isinstance(out, dict) else out[0]).float()
        all_feats.append(sam_feat)

        # ConvNeXt (3 scales: stage 1, 2, 3)
        img_tensors = []
        for p in images:
            arr = np.array(p.resize((384, 384)))
            img_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
        batch_t = torch.stack(img_tensors).to(self.device).to(self.dtype)
        with torch.no_grad():
            scales = self.convnext_model(batch_t)
        for s in scales:
            all_feats.append(s.float())

        return all_feats
