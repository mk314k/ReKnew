"""Projector-camera calibration from a projected chessboard."""

from dataclasses import dataclass

import cv2
import numpy as np

from reknew.compositing import Size
from reknew.light import GAMMA, estimate_reflectance
from reknew.projection import footprint_corners

INNER_CORNERS = (9, 6)
PATTERN_LEVEL = 0.4


@dataclass
class ChessboardPattern:
    image_lin: np.ndarray
    corners: np.ndarray  # projector px, row by row from top-left
    inner_corners: tuple[int, int]
    level: float


def make_chessboard_pattern(projector_size: Size, inner_corners=INNER_CORNERS,
                            level: float = PATTERN_LEVEL) -> ChessboardPattern:
    """
    Centered chessboard on a background of `level` brightness. Kept below full brightness
    so the camera doesn't saturate and the projector gain can still be measured.
    """
    pw, ph = projector_size
    cols, rows = inner_corners
    square = int(min(pw / (cols + 3), ph / (rows + 3)))
    x0 = (pw - (cols + 1) * square) // 2
    y0 = (ph - (rows + 1) * square) // 2

    gray = np.full((ph, pw), level, dtype=np.float32)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                gray[y0 + r * square:y0 + (r + 1) * square, x0 + c * square:x0 + (c + 1) * square] = 0.0

    xs = x0 + square * np.arange(1, cols + 1)
    ys = y0 + square * np.arange(1, rows + 1)
    corners = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return ChessboardPattern(np.dstack([gray] * 3), corners, inner_corners, level)


def detect_projector_homography(ambient_lin: np.ndarray, observed_lin: np.ndarray,
                                pattern: ChessboardPattern) -> np.ndarray | None:
    """
    Projector->camera homography from a frame showing the projected chessboard.
    Subtracting the projector-off frame and dividing by reflectance removes room texture
    and dark surfaces before detection.
    Assumes the projector is not rotated more than 90 degrees relative to the camera.
    """
    reflectance = np.maximum(estimate_reflectance(ambient_lin).mean(axis=2), 0.05)
    added = np.clip(observed_lin - ambient_lin, 0.0, None).mean(axis=2) / reflectance
    scale = max(float(np.percentile(added, 99)), 1e-4)
    gray = (np.clip(added / scale, 0.0, 1.0) ** (1.0 / GAMMA) * 255).astype(np.uint8)

    found, corners = cv2.findChessboardCornersSB(gray, pattern.inner_corners, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern.inner_corners, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    corners = corners.reshape(-1, 2)
    if corners[0].sum() > corners[-1].sum():
        corners = corners[::-1]
    homography, _ = cv2.findHomography(pattern.corners, corners, cv2.RANSAC, 3.0)
    return homography


def measure_projector_gain(ambient_lin: np.ndarray, observed_lin: np.ndarray, pattern: ChessboardPattern,
                           proj_to_cam: np.ndarray, min_reflectance: float = 0.15,
                           saturation: float = 0.95) -> float | None:
    """Projector brightness in camera units, measured on the lit chessboard squares."""
    h, w = ambient_lin.shape[:2]
    lit = cv2.warpPerspective(pattern.image_lin[:, :, 0], proj_to_cam, (w, h)) > 0.99 * pattern.level
    lit = cv2.erode(lit.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    reflectance = estimate_reflectance(ambient_lin)
    usable = lit & (reflectance.min(axis=2) > min_reflectance) & (observed_lin.max(axis=2) < saturation)
    if usable.sum() < 100:
        return None
    ratio = (observed_lin - ambient_lin)[usable] / reflectance[usable]
    return float(np.median(ratio)) / pattern.level


def footprint_error(proj_to_cam_a: np.ndarray, proj_to_cam_b: np.ndarray, projector_size: Size) -> float:
    """Mean distance in camera pixels between where two homographies put the projector corners."""
    a = footprint_corners(proj_to_cam_a, projector_size)
    b = footprint_corners(proj_to_cam_b, projector_size)
    return float(np.linalg.norm(a - b, axis=1).mean())
