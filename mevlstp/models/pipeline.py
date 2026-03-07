"""
MEVL-STP Full Pipeline.

Combines multi-encoder detection with VLM recognition into a unified
text spotting pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image

from .fpn import LateralFPN
from .pse_decoder import MultiScaleDecoder
from .local_features import LocalFeatureEncoder


class MEVLSTPDetector(nn.Module):
    """
    Detection module: FPN + Local Feature Encoder + PSE Decoder.

    Note: The frozen encoders are managed externally (not nn.Module parameters)
    to avoid serializing ~10GB of frozen weights in checkpoints.
    """

    def __init__(
        self,
        encoder_dims: List[int],
        local_channels: int,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.fpn = LateralFPN(encoder_dims, embed_dim)
        self.local_encoder = LocalFeatureEncoder(local_channels, embed_dim)
        self.decoder = MultiScaleDecoder(embed_dim * 2)  # FPN + local = 2×embed_dim

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        local_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: List of [B, C_i, H_i, W_i] from frozen encoders.
            local_features: [B, local_channels, H_local, W_local].

        Returns:
            Tuple of (main, aux_half, aux_quarter) prediction maps.
        """
        fpn_out = self.fpn(encoder_features)
        local_out = self.local_encoder(local_features)

        # Resize local features to match FPN spatial size
        local_out = F.interpolate(
            local_out,
            size=(fpn_out.shape[2], fpn_out.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        combined = torch.cat([fpn_out, local_out], dim=1)
        return self.decoder(combined)


class MEVLSTPPipeline:
    """
    Full end-to-end text spotting pipeline.

    Usage:
        pipeline = MEVLSTPPipeline(det_checkpoint, rec_adapter)
        results = pipeline.spot(image)
    """

    def __init__(
        self,
        det_checkpoint: str,
        rec_adapter: Optional[str] = None,
        device: str = "cuda",
        det_threshold: float = 0.5,
        min_area: int = 200,
    ):
        from .encoders import MultiEncoderExtractor
        from .local_features import TextLocalFeatures
        from .vlm_recognizer import load_vlm_with_adapter, recognize_crops, prepare_crop
        from ..utils.pse_decode import pse_decode

        self.device = device
        self.det_threshold = det_threshold
        self.min_area = min_area
        self._prepare_crop = prepare_crop
        self._recognize_crops = recognize_crops
        self._pse_decode = pse_decode

        # Initialize encoders
        self.encoder = MultiEncoderExtractor(device=device)
        self.local_extractor = TextLocalFeatures(size=64)

        # Load detection model
        dummy = Image.new("RGB", (512, 512))
        feat_list = self.encoder.extract([dummy])
        encoder_dims = [f.shape[1] for f in feat_list]

        self.detector = MEVLSTPDetector(
            encoder_dims, self.local_extractor.num_channels
        ).to(device)

        ckpt = torch.load(det_checkpoint, map_location=device)
        self.detector.load_state_dict(ckpt["model_state_dict"])
        self.detector.eval()

        # Load VLM recognizer
        if rec_adapter:
            self.vlm_model, self.vlm_processor = load_vlm_with_adapter(
                adapter_path=rec_adapter, device=device
            )
        else:
            self.vlm_model = None
            self.vlm_processor = None

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[np.ndarray]:
        """
        Detect text instances and return polygon contours.

        Args:
            image: Input PIL image.

        Returns:
            List of Nx2 polygon arrays.
        """
        img_resized = image.resize((512, 512))

        encoder_feats = self.encoder.extract([img_resized])
        local_feats = self.local_extractor.extract_batch([img_resized], self.device)

        main_pred, _, _ = self.detector(encoder_feats, local_feats)
        prob = torch.sigmoid(main_pred).squeeze().cpu().numpy()

        polygons = self._pse_decode(prob, self.det_threshold, self.min_area)

        # Scale polygons back to original image size
        sx = image.width / 512
        sy = image.height / 512
        scaled = []
        for poly in polygons:
            poly = poly.astype(np.float64)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            scaled.append(poly)

        return scaled

    def spot(self, image: Image.Image) -> List[Dict]:
        """
        Full text spotting: detect + recognize.

        Args:
            image: Input PIL image.

        Returns:
            List of dicts with 'polygon' and 'text' keys.
        """
        polygons = self.detect(image)

        if not polygons or self.vlm_model is None:
            return [{"polygon": p, "text": ""} for p in polygons]

        crops = [self._prepare_crop(image, p) for p in polygons]
        texts = self._recognize_crops(
            self.vlm_model, self.vlm_processor, crops, self.device
        )

        return [
            {"polygon": p, "text": t}
            for p, t in zip(polygons, texts)
        ]
