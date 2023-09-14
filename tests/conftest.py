from pathlib import Path

import cv2
import numpy as np
import pytest

from reknew.compositing import order_quad
from tests.scenery import CAMERA_SIZE, POSTER


@pytest.fixture
def room() -> np.ndarray:
    """Warm gray wall with a dark red poster, uint8 BGR."""
    w, h = CAMERA_SIZE
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    image[:, :, 0] = 170
    image[POSTER] = (60, 40, 150)
    noise = np.random.default_rng(0).normal(0, 4, image.shape)
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def textured_room() -> np.ndarray:
    """Room with enough random structure for ORB features."""
    rng = np.random.default_rng(1)
    w, h = CAMERA_SIZE
    image = np.full((h, w, 3), 128, dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(0, w - 40), rng.integers(0, h - 40)
        sw, sh = rng.integers(10, 60, size=2)
        image[y:y + sh, x:x + sw] = rng.integers(0, 255, size=3)
    return image


@pytest.fixture
def memory() -> np.ndarray:
    """640x360 BGRA test image: color gradients with light and dark stripes."""
    h, w = 360, 640
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    image = np.empty((h, w, 4), dtype=np.uint8)
    image[:, :, 0] = 255 * (1 - x) * np.ones_like(y)
    image[:, :, 1] = 255 * y * np.ones_like(x)
    image[:, :, 2] = 255 * x * (1 - y)
    image[:, :, 3] = 255
    image[40:60, :, :3] = 240
    image[300:320, :, :3] = 20
    return image


@pytest.fixture
def memory_path(tmp_path, memory) -> Path:
    path = tmp_path / "memories" / "memory.png"
    path.parent.mkdir()
    cv2.imwrite(str(path), memory)
    return path


@pytest.fixture
def overlay_corners() -> np.ndarray:
    return order_quad([(200, 120), (520, 110), (530, 380), (190, 390)])
