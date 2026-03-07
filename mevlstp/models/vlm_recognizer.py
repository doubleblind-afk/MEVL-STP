"""
VLM Text Recognition Module.

Uses Qwen3-VL-8B-Instruct with LoRA adaptation for scene text recognition
on polygon-cropped text instances.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def build_vlm(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device: str = "cuda",
):
    """
    Build the VLM recognizer with LoRA adapters.

    Args:
        model_id: HuggingFace model identifier.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        device: Target device.

    Returns:
        Tuple of (model, processor).
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_vlm_with_adapter(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    adapter_path: str = None,
    device: str = "cuda",
):
    """
    Load the VLM with a pretrained LoRA adapter for inference.

    Args:
        model_id: Base model identifier.
        adapter_path: Path to the saved LoRA adapter.
        device: Target device.

    Returns:
        Tuple of (model, processor).
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor


def prepare_crop(
    image: Image.Image,
    polygon: np.ndarray,
    padding: int = 5,
    canvas_size: int = 256,
) -> Image.Image:
    """
    Extract a polygon-masked crop from an image.

    Args:
        image: Source PIL image.
        polygon: Nx2 array of polygon vertices.
        padding: Pixel padding around bounding box.
        canvas_size: Output canvas size.

    Returns:
        Cropped and masked PIL image on white background.
    """
    import cv2

    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Create polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

    # Apply mask (white background)
    masked = img_np.copy()
    masked[mask == 0] = 255

    # Crop to bounding box with padding
    x_min = max(0, polygon[:, 0].min() - padding)
    x_max = min(w, polygon[:, 0].max() + padding)
    y_min = max(0, polygon[:, 1].min() - padding)
    y_max = min(h, polygon[:, 1].max() + padding)
    crop = masked[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Resize to canvas preserving aspect ratio
    crop_pil = Image.fromarray(crop)
    crop_pil.thumbnail((canvas_size, canvas_size), Image.LANCZOS)

    # Pad to square canvas
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    offset_x = (canvas_size - crop_pil.width) // 2
    offset_y = (canvas_size - crop_pil.height) // 2
    canvas.paste(crop_pil, (offset_x, offset_y))

    return canvas


@torch.no_grad()
def recognize_crops(
    model,
    processor,
    crops: List[Image.Image],
    device: str = "cuda",
) -> List[str]:
    """
    Recognize text from a list of cropped images.

    Args:
        model: VLM model (with merged LoRA).
        processor: VLM processor.
        crops: List of cropped PIL images.
        device: Target device.

    Returns:
        List of recognized text strings.
    """
    results = []
    prompt = "Read the text in this image. Output only the text, nothing else."

    for crop in crops:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[crop], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()
        results.append(generated)

    return results
