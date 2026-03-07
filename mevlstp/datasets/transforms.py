"""Data augmentation transforms for text detection training."""

import albumentations as A
import cv2


def get_train_transforms(img_size: int = 512):
    """Training augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 512):
    """Validation/test transform pipeline."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
