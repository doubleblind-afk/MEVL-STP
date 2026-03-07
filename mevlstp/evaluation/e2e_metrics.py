"""End-to-end text spotting evaluation metrics."""

import numpy as np
import editdistance
from typing import List, Dict, Tuple, Optional


def normalized_edit_distance(pred: str, gt: str) -> float:
    """Compute 1 - NED between predicted and ground truth text."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    ned = editdistance.eval(pred.lower(), gt.lower()) / max(len(gt), len(pred))
    return 1.0 - ned


def evaluate_e2e(
    predictions: List[List[Dict]],
    ground_truths: List[List[Dict]],
    iou_threshold: float = 0.5,
    lexicon: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate end-to-end text spotting.

    Args:
        predictions: Per-image list of dicts with 'polygon' and 'text'.
        ground_truths: Per-image list of dicts with 'polygon' and 'text'.
        iou_threshold: IoU threshold for polygon matching.
        lexicon: Optional lexicon for constrained evaluation.

    Returns:
        Tuple of (precision, recall, h_mean).
    """
    from .detection_metrics import polygon_iou

    total_tp = 0
    total_pred = 0
    total_gt = 0

    for preds, gts in zip(predictions, ground_truths):
        total_pred += len(preds)
        total_gt += len(gts)

        matched_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                iou = polygon_iou(pred["polygon"], gt["polygon"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:
                pred_text = pred["text"].lower().strip()
                gt_text = gts[best_idx]["text"].lower().strip()

                if lexicon:
                    # Find closest match in lexicon
                    best_match = min(
                        lexicon,
                        key=lambda w: editdistance.eval(pred_text, w.lower()),
                    )
                    pred_text = best_match.lower()

                if pred_text == gt_text:
                    total_tp += 1
                    matched_gt.add(best_idx)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    h_mean = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, h_mean
