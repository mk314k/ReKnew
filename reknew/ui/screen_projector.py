import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QScreen
from PyQt6.QtWidgets import QLabel

from reknew.light import linear_to_srgb
from reknew.projectors import black_frame
from reknew.ui.widgets import bgr_to_pixmap


class ScreenProjector:
    """A real projector connected as a display: shows the output fullscreen on that screen."""

    is_simulated = False

    def __init__(self, screen: QScreen):
        self._pixel_ratio = screen.devicePixelRatio()
        geometry = screen.geometry()
        self.size = (int(geometry.width() * self._pixel_ratio), int(geometry.height() * self._pixel_ratio))
        self.image_lin = black_frame(self.size)

        self._window = QLabel()
        self._window.setWindowTitle("ReKnew - Projector")
        self._window.setStyleSheet("background-color: #000;")
        self._window.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._window.setScreen(screen)
        self._window.move(geometry.topLeft())
        self._window.showFullScreen()

    def show(self, image_lin: np.ndarray) -> None:
        self.image_lin = image_lin
        pixmap = bgr_to_pixmap(linear_to_srgb(image_lin))
        pixmap.setDevicePixelRatio(self._pixel_ratio)
        self._window.setPixmap(pixmap)

    def close(self) -> None:
        self._window.close()
