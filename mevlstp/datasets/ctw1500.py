"""CTW1500 dataset loader for curved text detection."""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from typing import List, Dict, Tuple


def load_ctw1500(
    image_dir: str,
    label_dir: str,
) -> List[Dict]:
    """
    Load CTW1500 dataset.

    Args:
        image_dir: Path to image directory.
        label_dir: Path to XML label directory.

    Returns:
        List of dicts with 'image_path' and 'label_path'.
    """
    data = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        stem = os.path.splitext(f)[0]
        label_path = os.path.join(label_dir, f"{stem}.xml")
        if os.path.exists(label_path):
            data.append({
                "image_path": os.path.join(image_dir, f),
                "label_path": label_path,
            })
    return data


def parse_ctw1500_label(xml_path: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Parse CTW1500 XML annotation into a binary mask.

    Args:
        xml_path: Path to XML annotation file.
        shape: (H, W) of the target mask.

    Returns:
        Binary mask of shape (H, W).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    try:
        root = ET.parse(xml_path).getroot()
        for box in root.findall(".//box"):
            segs = box.find("segs")
            if segs is not None and segs.text:
                coords = [int(float(x)) for x in segs.text.strip().split(",")]
                if len(coords) >= 8:
                    pts = np.array(
                        list(zip(coords[::2], coords[1::2])), dtype=np.int32
                    )
                    cv2.fillPoly(mask, [pts], 255)
    except Exception:
        pass
    return mask


def parse_ctw1500_polygons(xml_path: str) -> List[np.ndarray]:
    """
    Parse CTW1500 XML annotation into polygon arrays.

    Args:
        xml_path: Path to XML annotation file.

    Returns:
        List of Nx2 polygon arrays.
    """
    polygons = []
    try:
        root = ET.parse(xml_path).getroot()
        for box in root.findall(".//box"):
            segs = box.find("segs")
            if segs is not None and segs.text:
                coords = [int(float(x)) for x in segs.text.strip().split(",")]
                if len(coords) >= 8:
                    pts = np.array(
                        list(zip(coords[::2], coords[1::2])), dtype=np.int32
                    )
                    polygons.append(pts)
    except Exception:
        pass
    return polygons
