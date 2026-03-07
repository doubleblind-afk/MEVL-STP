"""Visualization utilities for detection and spotting results."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional


def draw_polygons(
    image: Image.Image,
    polygons: List[np.ndarray],
    texts: Optional[List[str]] = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_size: int = 14,
) -> Image.Image:
    """
    Draw polygon detections on an image.

    Args:
        image: Input PIL image.
        polygons: List of Nx2 polygon arrays.
        texts: Optional list of recognized text strings.
        color: Polygon line color (RGB).
        thickness: Line thickness.
        font_size: Text label font size.

    Returns:
        Annotated PIL image.
    """
    img_np = np.array(image).copy()

    for i, poly in enumerate(polygons):
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_np, [pts], isClosed=True, color=color, thickness=thickness)

        if texts and i < len(texts) and texts[i]:
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min()) - 5
            cv2.putText(
                img_np, texts[i], (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    return Image.fromarray(img_np)
