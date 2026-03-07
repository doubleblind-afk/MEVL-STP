"""Total-Text dataset loader for curved and multi-oriented text detection."""

import os
import cv2
import numpy as np
import scipy.io as sio
from typing import List, Dict, Tuple


def load_totaltext(
    image_dir: str,
    label_dir: str,
) -> List[Dict]:
    """Load Total-Text dataset."""
    data = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        stem = os.path.splitext(f)[0]
        for candidate in [f"poly_gt_{stem}.mat", f"{stem}.mat", f"gt_{stem}.mat"]:
            label_path = os.path.join(label_dir, candidate)
            if os.path.exists(label_path):
                data.append({
                    "image_path": os.path.join(image_dir, f),
                    "label_path": label_path,
                })
                break
    return data


def parse_totaltext_label(mat_path: str, shape: Tuple[int, int]) -> np.ndarray:
    """Parse Total-Text .mat annotation into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    try:
        mat = sio.loadmat(mat_path)
        if "polygt" in mat:
            for row in mat["polygt"]:
                try:
                    xs = row[1].flatten().astype(int)
                    ys = row[3].flatten().astype(int)
                    if len(xs) >= 3:
                        pts = np.column_stack([xs, ys]).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                except Exception:
                    continue
    except Exception:
        pass
    return mask
