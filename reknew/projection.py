"""
Light transport between a projector and a camera.

A projector can only add light, so the camera sees

    observed = ambient + gain * reflectance * projector_light

where projector_light is the projector image mapped into camera space and gain is the
projector's brightness in camera units.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from reknew.compositing import Size
from reknew.light import estimate_reflectance, luminance


def footprint_corners(proj_to_cam: np.ndarray, projector_size: Size) -> np.ndarray:
    """Camera-space corners of the area the projector lights (top-left, clockwise)."""
    pw, ph = projector_size
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, proj_to_cam).reshape(4, 2)


def footprint_mask(proj_to_cam: np.ndarray, projector_size: Size, cam_size: Size) -> np.ndarray:
    """Boolean camera-space mask of the area the projector lights."""
    pw, ph = projector_size
    ones = np.ones((ph, pw), dtype=np.uint8)
    return cv2.warpPerspective(ones, proj_to_cam, cam_size, flags=cv2.INTER_NEAREST) > 0


def simulate_projection(ambient_lin: np.ndarray, projector_lin: np.ndarray, proj_to_cam: np.ndarray,
                        gain: float, reflectance: np.ndarray | None = None) -> np.ndarray:
    """Camera view of the room lit by its ambient light plus the projector (unclipped)."""
    h, w = ambient_lin.shape[:2]
    if reflectance is None:
        reflectance = estimate_reflectance(ambient_lin)
    light = cv2.warpPerspective(projector_lin, proj_to_cam, (w, h), flags=cv2.INTER_LINEAR)
    return ambient_lin + gain * reflectance * light


@dataclass(frozen=True)
class ContrastRange:
    """Darkest and brightest luminance the projector can produce across the overlay area."""

    black: float
    white: float

    @property
    def ratio(self) -> float:
        return self.white / max(self.black, 1e-3)


def compensation_target(content_lin: np.ndarray, coverage: np.ndarray, ambient_lin: np.ndarray,
                        reflectance: np.ndarray, gain: float,
                        uniformity: float = 0.5) -> tuple[np.ndarray, ContrastRange | None]:
    """
    Choose what the camera should see: content (composited over black) mapped into the
    brightness range the projector can produce on these surfaces, so it looks even across
    differently colored areas. The range runs from the brightest ambient level (a projector
    can't darken) to the dimmest achievable white.

    uniformity=1 fits every pixel (even but low contrast); lower values ignore the darkest
    and brightest surface patches, trading evenness for contrast.
    """
    region = coverage[::2, ::2] > 0
    if not region.any():
        return ambient_lin.copy(), None
    ambient_px = ambient_lin[::2, ::2][region]
    floor = luminance(ambient_px)
    ceiling = np.minimum(luminance(ambient_px + gain * reflectance[::2, ::2][region]), 1.0)

    percentile = 50.0 * (1.0 - float(np.clip(uniformity, 0.0, 1.0)))
    black = float(np.percentile(floor, 100.0 - percentile))
    white = max(float(np.percentile(ceiling, percentile)), black + 0.02)

    coverage_3 = coverage[:, :, None]
    target = ambient_lin * (1.0 - coverage_3) + black * coverage_3 + content_lin * (white - black)
    return target, ContrastRange(black, white)


def solve_projector_light(target_lin: np.ndarray, ambient_lin: np.ndarray, reflectance: np.ndarray,
                          gain: float, min_reflectance: float = 0.02,
                          tolerance: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve ambient + gain * reflectance * light = target for the projector light (camera space).
    Returns the light clipped to [0, 1] and a mask of pixels the projector can't reproduce:
    darker than the room, or brighter/more saturated than the surface can reflect.
    Wherever the target equals the ambient light, the light is zero.
    """
    light = (target_lin - ambient_lin) / (gain * np.maximum(reflectance, min_reflectance))
    b, g, r = light[:, :, 0], light[:, :, 1], light[:, :, 2]
    lowest = np.minimum(np.minimum(b, g), r)
    highest = np.maximum(np.maximum(b, g), r)
    unreachable = (lowest < -tolerance) | (highest > 1.0 + tolerance)
    return np.clip(light, 0.0, 1.0, out=light), unreachable
