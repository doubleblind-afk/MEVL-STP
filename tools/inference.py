"""
Run MEVL-STP inference on a single image or directory.

Usage:
    python tools/inference.py --image path/to/image.jpg \
        --det-checkpoint checkpoints/det_best.pth \
        --rec-checkpoint checkpoints/rec_adapter/ \
        --output results/
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevlstp.models.pipeline import MEVLSTPPipeline
from mevlstp.utils.visualize import draw_polygons


def main():
    parser = argparse.ArgumentParser(description="MEVL-STP Inference")
    parser.add_argument("--image", type=str, required=True, help="Image path or directory")
    parser.add_argument("--det-checkpoint", type=str, required=True)
    parser.add_argument("--rec-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing MEVL-STP pipeline...")
    pipeline = MEVLSTPPipeline(
        det_checkpoint=args.det_checkpoint,
        rec_adapter=args.rec_checkpoint,
        det_threshold=args.threshold,
    )

    # Collect images
    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = sorted([
            f for f in image_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
    else:
        image_files = [image_path]

    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        results = pipeline.spot(image)

        polygons = [r["polygon"] for r in results]
        texts = [r["text"] for r in results]

        # Visualize
        vis = draw_polygons(image, polygons, texts)
        out_path = output_dir / f"{img_path.stem}_result.jpg"
        vis.save(out_path)

        # Print results
        print(f"\n{img_path.name}: {len(results)} text instances")
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['text']}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
