"""Projector outputs: the interface every projector implements, and a simulated projector."""

from typing import Protocol

import cv2
import numpy as np

from reknew.compositing import Size
from reknew.light import estimate_reflectance, linear_to_srgb, srgb_to_linear
from reknew.projection import simulate_projection


class Projector(Protocol):
    is_simulated: bool
    size: Size
    image_lin: np.ndarray  # what the projector is currently showing

    def show(self, image_lin: np.ndarray) -> None: ...

    def close(self) -> None: ...


def black_frame(size: Size) -> np.ndarray:
    w, h = size
    return np.zeros((h, w, 3), dtype=np.float32)


class SimulatedProjector:
    """Virtual projector rigidly mounted next to the camera, offset so its image is keystoned."""

    is_simulated = True

    def __init__(self, size: Size = (1280, 720)):
        self.size = size
        self.image_lin = black_frame(size)
        self.true_proj_to_cam: np.ndarray | None = None

    @property
    def is_mounted(self) -> bool:
        return self.true_proj_to_cam is not None

    def mount_next_to_camera(self, cam_size: Size) -> None:
        w, h = cam_size
        pw, ph = self.size
        projector_corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        camera_corners = np.float32([
            [0.12 * w, 0.10 * h],
            [0.90 * w, 0.05 * h],
            [0.95 * w, 0.93 * h],
            [0.07 * w, 0.88 * h],
        ])
        self.true_proj_to_cam = cv2.getPerspectiveTransform(projector_corners, camera_corners)

    def show(self, image_lin: np.ndarray) -> None:
        self.image_lin = image_lin

    def capture(self, room_lin: np.ndarray, room_light: float, gain: float) -> np.ndarray:
        """What the camera would capture: the dimmed room plus projector light, clipped and quantized."""
        observed = simulate_projection(
            room_lin * room_light, self.image_lin, self.true_proj_to_cam, gain, estimate_reflectance(room_lin)
        )
        return srgb_to_linear(linear_to_srgb(observed))

    def close(self) -> None:
        pass
