"""Drawing helpers that annotate camera frames for display."""

import cv2
import numpy as np

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 200, 255)
MAGENTA = (255, 0, 255)


def draw_corners(frame: np.ndarray, corners) -> None:
    """Draw picked corners, numbered, joined into a closed polygon once all four are set."""
    if not corners:
        return
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], len(corners) == 4, GREEN, 2, cv2.LINE_AA)
    for index, (x, y) in enumerate(corners, start=1):
        cv2.circle(frame, (int(x), int(y)), 6, GREEN, -1)
        cv2.putText(frame, str(index), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def draw_outline(frame: np.ndarray, corners, color=ORANGE) -> None:
    cv2.polylines(frame, [np.asarray(corners, dtype=np.int32)], True, color, 1, cv2.LINE_AA)


def tint(frame: np.ndarray, mask: np.ndarray, color=MAGENTA, strength: float = 0.5) -> None:
    tinted = frame[mask].astype(np.float32) * (1 - strength) + np.float32(color) * strength
    frame[mask] = tinted.astype(np.uint8)
