"""
Progressive Scale Expansion (PSE) Post-Processing.

Decodes the network's probability map into individual text instance polygons
using breadth-first expansion from seed kernels.
"""

import cv2
import numpy as np
from typing import List
from collections import deque


def pse_decode(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 200,
    epsilon_ratio: float = 0.002,
) -> List[np.ndarray]:
    """
    Decode a probability map into text polygons using PSE.

    Args:
        prob_map: 2D array of text confidence values in [0, 1].
        threshold: Binarization threshold.
        min_area: Minimum contour area in pixels.
        epsilon_ratio: Douglas-Peucker simplification ratio.

    Returns:
        List of Nx2 polygon arrays.
    """
    binary = (prob_map > threshold).astype(np.uint8)

    # Connected components as seed instances
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    polygons = []
    for label_id in range(1, num_labels):
        instance_mask = (labels == label_id).astype(np.uint8)
        area = instance_mask.sum()
        if area < min_area:
            continue

        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue

        # Simplify polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            polygon = approx.squeeze()
            if polygon.ndim == 2 and polygon.shape[0] >= 3:
                polygons.append(polygon)

    return polygons


def pse_multi_kernel_decode(
    kernel_maps: np.ndarray,
    threshold_s1: float = 0.4,
    threshold_sk: float = 0.5,
    min_area: int = 50,
    epsilon_ratio: float = 0.002,
    min_contour_area: int = 200,
) -> List[np.ndarray]:
    """
    Full PSE decoding with K progressive kernels.

    The smallest kernel (index 0) is thresholded and labelled via connected
    components to produce seed instances. Each subsequent kernel expands
    seeds via BFS while preserving instance boundaries.

    Args:
        kernel_maps: Array of shape [K, H, W] with confidence values.
        threshold_s1: Threshold for the smallest kernel (seeds).
        threshold_sk: Threshold for subsequent kernels.
        min_area: Minimum seed area.
        epsilon_ratio: Douglas-Peucker simplification ratio.
        min_contour_area: Minimum final polygon area.

    Returns:
        List of Nx2 polygon arrays.
    """
    K, H, W = kernel_maps.shape

    # Step 1: Threshold smallest kernel to get seeds
    seed_map = (kernel_maps[0] > threshold_s1).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(seed_map, connectivity=8)

    # Step 2: Progressive expansion through larger kernels
    for k in range(1, K):
        expansion_mask = (kernel_maps[k] > threshold_sk).astype(np.uint8)
        new_labels = labels.copy()

        # BFS expansion
        queue = deque()
        for y in range(H):
            for x in range(W):
                if labels[y, x] > 0:
                    queue.append((y, x))

        dy = [-1, 0, 1, 0]
        dx = [0, 1, 0, -1]

        while queue:
            cy, cx = queue.popleft()
            current_label = new_labels[cy, cx]
            for d in range(4):
                ny, nx = cy + dy[d], cx + dx[d]
                if 0 <= ny < H and 0 <= nx < W:
                    if new_labels[ny, nx] == 0 and expansion_mask[ny, nx] == 1:
                        new_labels[ny, nx] = current_label
                        queue.append((ny, nx))

        labels = new_labels

    # Step 3: Extract polygons from final label map
    polygons = []
    for label_id in range(1, num_labels):
        instance_mask = (labels == label_id).astype(np.uint8)
        if instance_mask.sum() < min_area:
            continue

        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_contour_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            polygon = approx.squeeze()
            if polygon.ndim == 2 and polygon.shape[0] >= 3:
                polygons.append(polygon)

    return polygons
