# Installation Guide

## Prerequisites

- Python >= 3.10
- CUDA >= 11.8
- ~40GB GPU memory (for all six encoders)

## Step 1: Create environment

```bash
conda create -n mevlstp python=3.10
conda activate mevlstp
```

## Step 2: Install PyTorch

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install dependencies

```bash
cd MEVL-STP
pip install -r requirements.txt
pip install -e .
```

## Step 4: Download pretrained encoders

The six frozen encoders are automatically downloaded from HuggingFace on first run:
- `openai/clip-vit-large-patch14-336`
- `facebook/dinov2-large`
- `google/siglip-so400m-patch14-384`
- `openai/clip-vit-large-patch14`
- `facebook/sam-vit-huge`
- `convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384` (via timm)

## Step 5: Prepare datasets

See the [README](../README.md) for dataset download links and directory structure.

## Memory Requirements

| Configuration | GPU Memory |
|---------------|-----------|
| Inference (all 6 encoders) | ~40 GB |
| Training (detection only) | ~45 GB |
| Training (E2E with VLM) | ~70 GB |

For GPUs with less memory, encoders can be loaded sequentially rather than in parallel.
