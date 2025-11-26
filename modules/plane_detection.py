import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_from_contours(contours, min_area):
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) >= min_area:
            quad = approx.reshape(4, 2).astype("float32")
            return _order_points(quad)
    return None


def _edges_based_quad(frame: np.ndarray, min_area: float):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return _quad_from_contours(contours, min_area)


def _mediapipe_mask_quad(frame: np.ndarray, min_area: float):
    if not _HAS_MEDIAPIPE:
        return None
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_selfie.process(rgb)
    mask = (result.segmentation_mask * 255).astype(np.uint8)
    mp_selfie.close()
    mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return _quad_from_contours(contours, min_area)


def suggest_quad(frame: np.ndarray, min_area: float = 5000.0):
    """
    Heuristic plane/edge detection: try segmentation-assisted quad, then edges.
    """
    quad = _mediapipe_mask_quad(frame, min_area)
    if quad is not None:
        return quad
    return _edges_based_quad(frame, min_area)
