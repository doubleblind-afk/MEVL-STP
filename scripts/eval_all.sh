#!/bin/bash
# Evaluate MEVL-STP detection on all three benchmarks

echo "=== CTW1500 ==="
python tools/eval_detection.py --config configs/ctw1500_det.yaml --checkpoint $1

echo "=== Total-Text ==="
python tools/eval_detection.py --config configs/totaltext_det.yaml --checkpoint $1

echo "=== ICDAR 2015 ==="
python tools/eval_detection.py --config configs/icdar2015_det.yaml --checkpoint $1
