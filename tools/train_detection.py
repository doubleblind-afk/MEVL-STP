"""
Train the MEVL-STP detection stage.

Usage:
    python tools/train_detection.py --config configs/ctw1500_det.yaml
    python tools/train_detection.py --config configs/totaltext_det.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevlstp.models.encoders import MultiEncoderExtractor
from mevlstp.models.local_features import TextLocalFeatures
from mevlstp.models.pipeline import MEVLSTPDetector
from mevlstp.datasets.ctw1500 import load_ctw1500, parse_ctw1500_label
from mevlstp.datasets.totaltext import load_totaltext, parse_totaltext_label
from mevlstp.datasets.icdar2015 import load_icdar2015, parse_icdar2015_label


def focal_loss(pred, target, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.exp(-bce)
    return (alpha * (1 - p_t) ** gamma * bce).mean()


def dice_loss(pred, target, smooth=1.0):
    p = torch.sigmoid(pred)
    num = 2 * (p * target).sum() + smooth
    den = p.sum() + target.sum() + smooth
    return 1 - num / den


def ohem_bce_loss(pred, target, keep_ratio=0.7):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    n_keep = max(int(keep_ratio * bce.numel()), 1)
    hard, _ = bce.view(-1).topk(n_keep)
    return hard.mean()


def combined_loss(pred, target):
    return 0.5 * focal_loss(pred, target) + 0.5 * dice_loss(pred, target) + 0.3 * ohem_bce_loss(pred, target)


def deep_sup_loss(main, aux2, aux1, target):
    h, w = main.shape[2], main.shape[3]
    t_full = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
    t_half = F.interpolate(target, size=(aux2.shape[2], aux2.shape[3]), mode="bilinear", align_corners=False)
    t_qtr = F.interpolate(target, size=(aux1.shape[2], aux1.shape[3]), mode="bilinear", align_corners=False)
    return 1.0 * combined_loss(main, t_full) + 0.4 * combined_loss(aux2, t_half) + 0.2 * combined_loss(aux1, t_qtr)


def main():
    parser = argparse.ArgumentParser(description="Train MEVL-STP detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/detection")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("=" * 60)
    print(f"MEVL-STP Detection Training — {cfg['data']['dataset']}")
    print(f"Output: {save_dir}")
    print("=" * 60)

    # Load data
    dataset = cfg["data"]["dataset"]
    if dataset != "ctw1500":
        print(f"ERROR: Currently only CTW1500 is supported. Got dataset={dataset}")
        print("For Total-Text and ICDAR 2015 results, see the paper.")
        sys.exit(1)

    if dataset == "ctw1500":
        data = load_ctw1500(cfg["data"]["train_images"], cfg["data"]["train_labels"])
        parse_fn = parse_ctw1500_label
    elif dataset == "totaltext":
        data = load_totaltext(cfg["data"]["train_images"], cfg["data"]["train_labels"])
        parse_fn = parse_totaltext_label
    elif dataset == "icdar2015":
        data = load_icdar2015(cfg["data"]["train_images"], cfg["data"]["train_labels"])
        parse_fn = parse_icdar2015_label
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Loaded {len(data)} training samples")

    # Initialize encoders
    print("Loading frozen encoders...")
    img_size = cfg["model"]["img_size"]
    encoder = MultiEncoderExtractor(device=str(device))
    local_extractor = TextLocalFeatures(size=cfg["model"]["local_feat_size"])

    # Probe encoder dimensions
    dummy = Image.new("RGB", (img_size, img_size))
    feat_list = encoder.extract([dummy])
    encoder_dims = [f.shape[1] for f in feat_list]
    print(f"Encoder dims: {encoder_dims}")

    # Build model
    detector = MEVLSTPDetector(encoder_dims, local_extractor.num_channels).to(device)
    print(f"Trainable params: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"]["warmup_epochs"]
    accum_steps = cfg["train"]["accum_steps"]
    batch_size = cfg["train"]["batch_size"]

    # EMA
    ema_model = copy.deepcopy(detector)
    ema_decay = cfg["train"]["ema_decay"]

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        detector.train()
        np.random.shuffle(data)
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, start in enumerate(pbar):
            batch_data = data[start : start + batch_size]
            images, masks = [], []

            for sample in batch_data:
                img = Image.open(sample["image_path"]).convert("RGB").resize((img_size, img_size))
                mask = parse_fn(sample["label_path"], (img_size, img_size))
                images.append(img)
                masks.append(torch.from_numpy(mask).float().unsqueeze(0) / 255.0)

            if not images:
                continue

            mask_batch = torch.stack(masks).to(device)

            with torch.no_grad():
                enc_feats = encoder.extract(images)
                local_feats = local_extractor.extract_batch(images, str(device))

            main_pred, aux2, aux1 = detector(enc_feats, local_feats)
            loss = deep_sup_loss(main_pred, aux2, aux1, mask_batch) / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                with torch.no_grad():
                    for p_ema, p_model in zip(ema_model.parameters(), detector.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1 - ema_decay)

            epoch_loss += loss.item() * accum_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, save_dir / "best_model.pth")
            print(f"  Saved best model (loss={avg_loss:.4f})")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model.state_dict(),
                "loss": avg_loss,
            }, save_dir / f"epoch_{epoch}.pth")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
