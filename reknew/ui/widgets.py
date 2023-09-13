from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QMimeData, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QSlider, QWidget

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _thumbnail(path: str, width: int, height: int) -> QPixmap:
    return QPixmap(path).scaled(
        width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class MemoryList(QListWidget):
    """Thumbnails of memory images; drag one onto the camera view or double-click it."""

    memoryActivated = pyqtSignal(str)

    def __init__(self, image_dir: Path, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(140, 100))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(8)
        self.setDragEnabled(True)
        self.itemDoubleClicked.connect(lambda item: self.memoryActivated.emit(self._path_of(item)))
        if image_dir.exists():
            for path in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.add_memory(str(path))

    def add_memory(self, path: str) -> None:
        item = QListWidgetItem(Path(path).stem)
        item.setIcon(QIcon(_thumbnail(path, 160, 120)))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.addItem(item)

    @staticmethod
    def _path_of(item: QListWidgetItem) -> str:
        return item.data(Qt.ItemDataRole.UserRole)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        path = self._path_of(item)
        mime = QMimeData()
        mime.setText(path)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(_thumbnail(path, 180, 120))
        drag.exec(Qt.DropAction.CopyAction)


class CameraView(QLabel):
    """Shows camera frames at native size; reports clicks in frame pixels and accepts image drops."""

    clicked = pyqtSignal(int, int)
    imageDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #000;")
        self.setMinimumSize(640, 480)

    def show_frame(self, frame: np.ndarray) -> None:
        pixmap = bgr_to_pixmap(frame)
        if self.size() != pixmap.size():
            self.setFixedSize(pixmap.size())
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().text().removeprefix("file://")
        if path:
            self.imageDropped.emit(path)
        event.acceptProposedAction()


class LabeledSlider(QWidget):
    """Horizontal slider over a float range, labeled with its current value."""

    valueChanged = pyqtSignal(float)

    _STEPS_PER_UNIT = 100

    def __init__(self, name: str, minimum: float, maximum: float, value: float, tooltip: str = ""):
        super().__init__()
        self._name = name
        self._label = QLabel()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(self._to_steps(minimum), self._to_steps(maximum))
        self._slider.setValue(self._to_steps(value))
        self._slider.valueChanged.connect(self._on_slider_moved)
        if tooltip:
            self.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self._slider)
        self._update_label()

    def value(self) -> float:
        return self._slider.value() / self._STEPS_PER_UNIT

    def set_value(self, value: float) -> None:
        self._slider.setValue(self._to_steps(value))

    def _to_steps(self, value: float) -> int:
        return round(value * self._STEPS_PER_UNIT)

    def _update_label(self) -> None:
        self._label.setText(f"{self._name}: {self.value():.2f}")

    def _on_slider_moved(self, _steps: int) -> None:
        self._update_label()
        self.valueChanged.emit(self.value())
