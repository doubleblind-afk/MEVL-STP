"""
Train the MEVL-STP VLM recognition stage (LoRA fine-tuning).

Usage:
    python tools/train_recognition.py --config configs/ctw1500_rec.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevlstp.models.vlm_recognizer import build_vlm, prepare_crop


def main():
    parser = argparse.ArgumentParser(description="Train MEVL-STP VLM recognition")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/recognition")
    parser.add_argument("--crop-dir", type=str, required=True, help="Directory of polygon crops")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MEVL-STP VLM Recognition Training (LoRA)")
    print(f"Output: {save_dir}")
    print("=" * 60)

    # Build model with LoRA
    lora_cfg = cfg["model"]["lora"]
    model, processor = build_vlm(
        model_id=cfg["model"]["vlm_model_id"],
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        device=device,
    )

    # Training setup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
    )

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]

    print(f"LoRA config: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}")
    print(f"Training for {epochs} epochs, batch_size={batch_size}")

    # Load crop data
    crop_files = sorted([
        f for f in os.listdir(args.crop_dir)
        if f.endswith((".jpg", ".png"))
    ])
    print(f"Loaded {len(crop_files)} crops")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(crop_files)
        epoch_loss = 0.0
        n_steps = 0

        pbar = tqdm(range(0, len(crop_files), batch_size), desc=f"Epoch {epoch}/{epochs}")

        for start in pbar:
            batch_files = crop_files[start : start + batch_size]
            # Process batch (simplified — actual implementation loads crop + GT text pairs)
            # ...
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # Save adapter
        adapter_dir = save_dir / f"adapter_epoch_{epoch}"
        model.save_pretrained(str(adapter_dir))
        print(f"  Saved adapter to: {adapter_dir}")

    print(f"\nTraining complete. Adapters saved to: {save_dir}")


if __name__ == "__main__":
    main()
