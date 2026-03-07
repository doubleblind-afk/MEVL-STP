"""
Evaluate MEVL-STP end-to-end text spotting.

Usage:
    python tools/eval_e2e.py --config configs/ctw1500_e2e.yaml \
        --det-checkpoint checkpoints/det_best.pth \
        --rec-checkpoint checkpoints/rec_adapter/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mevlstp.models.pipeline import MEVLSTPPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate MEVL-STP end-to-end")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--det-checkpoint", type=str, required=True)
    parser.add_argument("--rec-checkpoint", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pipeline = MEVLSTPPipeline(
        det_checkpoint=args.det_checkpoint,
        rec_adapter=args.rec_checkpoint,
    )

    print("End-to-end evaluation pipeline initialized.")
    print("Use MEVLSTPPipeline.spot(image) to run inference.")


if __name__ == "__main__":
    main()
