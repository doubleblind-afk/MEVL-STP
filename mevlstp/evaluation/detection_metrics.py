"""Detection evaluation metrics (Precision, Recall, F1 at IoU >= 0.5)."""

import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute IoU between two polygons."""
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_detection(
    pred_polygons: List[List[np.ndarray]],
    gt_polygons: List[List[np.ndarray]],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Evaluate detection performance across a dataset.

    Args:
        pred_polygons: List of (list of predicted polygons) per image.
        gt_polygons: List of (list of ground truth polygons) per image.
        iou_threshold: IoU threshold for a match.

    Returns:
        Tuple of (precision, recall, f1).
    """
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for preds, gts in zip(pred_polygons, gt_polygons):
        total_pred += len(preds)
        total_gt += len(gts)

        matched_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                iou = polygon_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                total_tp += 1
                matched_gt.add(best_idx)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
