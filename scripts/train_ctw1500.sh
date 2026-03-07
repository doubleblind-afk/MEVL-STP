#!/bin/bash
# Train MEVL-STP detection on CTW1500
python tools/train_detection.py --config configs/ctw1500_det.yaml --output-dir output/ctw1500_det
