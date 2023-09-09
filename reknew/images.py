from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_MEMORY_SIDE = 1024


def load_bgra(path: str | Path) -> np.ndarray:
    rgba = Image.open(path).convert("RGBA")
    return cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)


def limit_size(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def load_memory_image(path: str | Path, max_side: int = MAX_MEMORY_SIDE) -> np.ndarray:
    """Load a memory as BGRA, downscaled so per-frame warping stays cheap."""
    return limit_size(load_bgra(path), max_side)
