"""ICDAR 2015 dataset loader for multi-oriented text detection."""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple


def load_icdar2015(
    image_dir: str,
    label_dir: str,
) -> List[Dict]:
    """Load ICDAR 2015 dataset."""
    data = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        stem = os.path.splitext(f)[0]
        for candidate in [f"gt_{stem}.txt", f"{stem}.txt"]:
            label_path = os.path.join(label_dir, candidate)
            if os.path.exists(label_path):
                data.append({
                    "image_path": os.path.join(image_dir, f),
                    "label_path": label_path,
                })
                break
    return data


def parse_icdar2015_label(txt_path: str, shape: Tuple[int, int]) -> np.ndarray:
    """Parse ICDAR 2015 text annotation into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    try:
        with open(txt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",", 8)
                if len(parts) >= 8:
                    try:
                        coords = [int(parts[i]) for i in range(8)]
                        pts = np.array([
                            [coords[0], coords[1]],
                            [coords[2], coords[3]],
                            [coords[4], coords[5]],
                            [coords[6], coords[7]],
                        ], dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    return mask
