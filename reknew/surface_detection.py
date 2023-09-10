"""Heuristic suggestion of a four-corner surface in a camera frame."""

import cv2
import numpy as np

from reknew.compositing import order_quad


def suggest_quad(frame: np.ndarray, min_area: float = 5000.0) -> np.ndarray | None:
    """Corners of the largest four-sided outline found in the frame's edges, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) >= min_area:
            return order_quad(approx.reshape(4, 2))
    return None
