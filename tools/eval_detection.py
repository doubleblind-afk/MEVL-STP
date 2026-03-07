"""
Evaluate MEVL-STP detection on a benchmark dataset.

Usage:
    python tools/eval_detection.py --config configs/ctw1500_det.yaml --checkpoint checkpoints/det_best.pth
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevlstp.models.encoders import MultiEncoderExtractor
from mevlstp.models.local_features import TextLocalFeatures
from mevlstp.models.pipeline import MEVLSTPDetector
from mevlstp.utils.pse_decode import pse_decode
from mevlstp.evaluation.detection_metrics import evaluate_detection
from mevlstp.datasets.ctw1500 import load_ctw1500, parse_ctw1500_polygons


def main():
    parser = argparse.ArgumentParser(description="Evaluate MEVL-STP detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cfg["model"]["img_size"]

    print("Loading model...")
    encoder = MultiEncoderExtractor(device=str(device))
    local_extractor = TextLocalFeatures(size=cfg["model"]["local_feat_size"])

    dummy = Image.new("RGB", (img_size, img_size))
    feat_list = encoder.extract([dummy])
    encoder_dims = [f.shape[1] for f in feat_list]

    detector = MEVLSTPDetector(encoder_dims, local_extractor.num_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    detector.load_state_dict(ckpt["model_state_dict"])
    detector.eval()

    # Load test data
    test_data = load_ctw1500(cfg["data"]["test_images"], cfg["data"]["test_labels"])
    if args.limit > 0:
        test_data = test_data[: args.limit]
    print(f"Evaluating on {len(test_data)} images...")

    all_preds = []
    all_gts = []

    for sample in tqdm(test_data):
        img = Image.open(sample["image_path"]).convert("RGB")
        img_resized = img.resize((img_size, img_size))

        with torch.no_grad():
            enc_feats = encoder.extract([img_resized])
            local_feats = local_extractor.extract_batch([img_resized], str(device))
            main_pred, _, _ = detector(enc_feats, local_feats)

        prob = torch.sigmoid(main_pred).squeeze().cpu().numpy()
        pred_polys = pse_decode(prob, args.threshold)

        # Scale to original image size
        sx, sy = img.width / img_size, img.height / img_size
        scaled_preds = []
        for p in pred_polys:
            p = p.astype(np.float64)
            p[:, 0] *= sx
            p[:, 1] *= sy
            scaled_preds.append(p)

        gt_polys = parse_ctw1500_polygons(sample["label_path"])

        all_preds.append(scaled_preds)
        all_gts.append(gt_polys)

    precision, recall, f1 = evaluate_detection(all_preds, all_gts)
    print(f"\nResults:")
    print(f"  Precision: {precision * 100:.1f}%")
    print(f"  Recall:    {recall * 100:.1f}%")
    print(f"  F1:        {f1 * 100:.1f}%")


if __name__ == "__main__":
    main()
