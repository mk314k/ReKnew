# modules/object_selection.py

import cv2
import numpy as np

# Track up to 4 corner points for perspective selection
points = []
selection_complete = False


def click_and_select(event, x, y, flags, param):
    """Collect four corner clicks; ignore extras until reset."""
    global points, selection_complete

    if event == cv2.EVENT_LBUTTONDOWN:
        if selection_complete:
            return
        points.append((x, y))
        if len(points) == 4:
            selection_complete = True


def reset_selection():
    global points, selection_complete
    points = []
    selection_complete = False


def draw_selection(image):
    """Visualize the current point picks and polygon."""
    if not points:
        return

    pts = np.array(points, dtype=np.int32)
    # Draw lines between selected points
    cv2.polylines(
        image,
        [pts],
        len(points) == 4,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # Draw small circles/labels for each point
    for idx, (px, py) in enumerate(points, start=1):
        cv2.circle(image, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(
            image,
            str(idx),
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
