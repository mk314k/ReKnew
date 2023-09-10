"""Warping memory images onto four-corner quads and compositing them."""

from dataclasses import dataclass

import cv2
import numpy as np

from reknew.light import srgb_to_linear

Size = tuple[int, int]  # (width, height)


def order_quad(points) -> np.ndarray:
    """Order four points clockwise starting at top-left, so overlays are never mirrored."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def warp_to_quad(image: np.ndarray, quad, out_size: Size) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-warp a BGR/BGRA image onto a quad (top-left, clockwise).
    Returns the warped BGR uint8 image and its alpha as float32 in [0, 1].
    """
    h, w = image.shape[:2]
    source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    transform = cv2.getPerspectiveTransform(source, np.float32(quad).reshape(4, 2))

    warped = cv2.warpPerspective(image[:, :, :3], transform, out_size)
    if image.shape[2] == 4:
        alpha_source = image[:, :, 3]
    else:
        alpha_source = np.full((h, w), 255, dtype=np.uint8)
    alpha = cv2.warpPerspective(alpha_source, transform, out_size).astype(np.float32) / 255.0
    return warped, alpha


@dataclass
class Layer:
    image: np.ndarray
    quad: np.ndarray
    opacity: float = 1.0
    brightness: float = 1.0


def composite_layers(layers: list[Layer], out_size: Size) -> tuple[np.ndarray, np.ndarray]:
    """
    Alpha-composite layers over black in linear light.
    Returns the composite and the combined coverage alpha (H x W, float32).
    """
    w, h = out_size
    result = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)
    for layer in layers:
        bgr, alpha = warp_to_quad(layer.image, layer.quad, out_size)
        alpha *= layer.opacity
        alpha_3 = alpha[:, :, None]
        result = result * (1.0 - alpha_3) + srgb_to_linear(bgr) * layer.brightness * alpha_3
        coverage += alpha * (1.0 - coverage)
    return result, coverage
