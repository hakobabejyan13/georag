import numpy as np
from PIL import Image

# Per-band normalization percentiles from dataset stats.
P1 = {
    1: 38,  # Blue B02
    2: 70,  # Green B03
    3: 36,  # Red B04
}
P99 = {
    1: 2291,
    2: 2589,
    3: 3140,
}


def extract_rgb(arr13: np.ndarray) -> np.ndarray:
    """
    Extract Sentinel-2 RGB channels from (13, H, W) array.
    Band 2 (Blue) -> index 1
    Band 3 (Green) -> index 2
    Band 4 (Red) -> index 3
    """
    if arr13.ndim != 3 or arr13.shape[0] < 4:
        raise ValueError(f"Expected array shape (>=4, H, W), got {arr13.shape}")

    red = arr13[3]
    green = arr13[2]
    blue = arr13[1]
    return np.stack([red, green, blue], axis=0)


def normalize_percentile(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Apply per-band percentile normalization to (3, H, W) RGB data.
    """
    if rgb_arr.ndim != 3 or rgb_arr.shape[0] != 3:
        raise ValueError(f"Expected RGB shape (3, H, W), got {rgb_arr.shape}")

    red = rgb_arr[0]
    green = rgb_arr[1]
    blue = rgb_arr[2]

    red_n = (red - P1[3]) / (P99[3] - P1[3])
    green_n = (green - P1[2]) / (P99[2] - P1[2])
    blue_n = (blue - P1[1]) / (P99[1] - P1[1])

    rgb_norm = np.stack([red_n, green_n, blue_n], axis=0)
    return np.clip(rgb_norm, 0, 1)


def to_pil(rgb_arr: np.ndarray) -> Image.Image:
    """
    Convert normalized (3, H, W) array to PIL Image (H, W, 3).
    """
    if rgb_arr.ndim != 3 or rgb_arr.shape[0] != 3:
        raise ValueError(f"Expected RGB shape (3, H, W), got {rgb_arr.shape}")

    rgb_uint8 = (rgb_arr * 255).astype(np.uint8)
    rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))
    return Image.fromarray(rgb_uint8)
