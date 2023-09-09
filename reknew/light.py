"""
Color conversion and surface estimates in linear light.

Arrays named `*_lin` are float32 BGR images in linear light, where 1.0 is camera saturation.
"""

import numpy as np

GAMMA = 2.2
_ENCODE_LEVELS = 4096
_DECODE_LUT = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)
# Indexed by sqrt(linear) so dark values keep enough resolution.
_ENCODE_LUT = np.round(np.linspace(0.0, 1.0, _ENCODE_LEVELS) ** (2.0 / GAMMA) * 255).astype(np.uint8)
_LUMA_WEIGHTS_BGR = np.float32([0.0722, 0.7152, 0.2126])


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """uint8 sRGB image to float32 linear light."""
    return _DECODE_LUT[image]


def linear_to_srgb(image_lin: np.ndarray) -> np.ndarray:
    """Linear light to uint8 sRGB, clipping to [0, 1]."""
    index = (np.sqrt(np.clip(image_lin, 0.0, 1.0)) * (_ENCODE_LEVELS - 1) + 0.5).astype(np.uint16)
    return _ENCODE_LUT[index]


def luminance(image_lin: np.ndarray) -> np.ndarray:
    return image_lin @ _LUMA_WEIGHTS_BGR


def estimate_reflectance(ambient_lin: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Per-pixel surface reflectance in [0, 1], normalized so the brightest surfaces are ~1."""
    sample = ambient_lin[::4, ::4].reshape(-1, 3)
    white = np.percentile(sample, percentile, axis=0)
    inverse_white = (1.0 / np.maximum(white, 1e-4)).astype(np.float32)
    return np.minimum(ambient_lin * inverse_white, 1.0)
