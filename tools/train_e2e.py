"""
End-to-end phased training for MEVL-STP.

Three-phase schedule:
  Phase 1 (epochs 1-2):   Detection only
  Phase 2 (epochs 3-10):  Sparse VLM supervision
  Phase 3 (epochs 11-30): Full VLM supervision

Usage:
    python tools/train_e2e.py --config configs/ctw1500_e2e.yaml
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Train MEVL-STP end-to-end")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/e2e")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("MEVL-STP End-to-End Phased Training")
    print("=" * 60)

    phases = cfg.get("e2e", {}).get("phases", [])
    for phase in phases:
        print(f"\n--- {phase['name']} (epochs {phase['epochs']}) ---")
        print(f"  lambda_rec: {phase['lambda_rec']}")

    print("\nSee tools/train_detection.py and tools/train_recognition.py")
    print("for the individual stage implementations.")


if __name__ == "__main__":
    main()
