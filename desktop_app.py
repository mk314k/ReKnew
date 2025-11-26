import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QMimeData, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDrag, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from modules.image_overlay import overlay_image
from modules.utils import load_image
from modules.plane_detection import suggest_quad
from modules.scene_persistence import compute_features, reproject_quad, save_scene, load_scene


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert BGR numpy frame to QPixmap."""
    h, w, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image.copy())


def draw_selection(frame: np.ndarray, points):
    """Draw polygon and indices for current selection."""
    if not points:
        return frame
    vis = frame.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(
        vis,
        [pts],
        len(points) == 4,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            vis,
            str(idx),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def _clean_path(path: str) -> str:
    return path[7:] if path.startswith("file://") else path


class MemoryList(QListWidget):
    """Thumbnail list of memory images that can be dragged onto the camera view."""

    memoryActivated = pyqtSignal(str)

    def __init__(self, image_dir: Path, parent=None):
        super().__init__(parent)
        self.image_dir = image_dir
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(140, 100))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(8)
        self.setDragEnabled(True)
        self.load_images()
        self.itemDoubleClicked.connect(self._on_double_click)

    def load_images(self):
        self.clear()
        if not self.image_dir.exists():
            return
        for path in sorted(
            self.image_dir.glob("*.*"),
            key=lambda p: p.name.lower(),
        ):
            self._add_item(path)

    def _add_item(self, path: Path):
        pixmap = QPixmap(str(path)).scaled(
            160, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        item = QListWidgetItem(path.stem)
        item.setIcon(QIcon(pixmap))
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.addItem(item)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        mime = QMimeData()
        mime.setText(path)
        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = QPixmap(path).scaled(
            180, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        drag.setPixmap(pixmap)
        drag.exec(Qt.DropAction.CopyAction)

    def _on_double_click(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.memoryActivated.emit(path)


class CameraView(QLabel):
    """Displays camera frames, handles quad clicks, and accepts drops."""

    pointClicked = pyqtSignal(tuple)
    imageDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #000;")
        self.setMinimumSize(640, 480)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self.pointClicked.emit((pos.x(), pos.y()))
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().text()
        if path:
            self.imageDropped.emit(path)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReKnew - Desktop Memory Overlay")

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise IOError("Cannot open webcam")

        self.current_frame = None
        self.mixed_frame = None
        self.points = []
        self.paused = False
        self.show_mixed = False
        self.match_colors = True
        self.default_opacity = 1.0
        self.default_brightness = 1.0
        self.overlay_records = []  # list of dicts: overlay_img, base_quad, base_kp, base_des

        self.camera_view = CameraView()
        self.status_label = QLabel("Click 4 corners of a surface.")

        memories_dir = Path("assets/images")
        self.memory_list = MemoryList(memories_dir)
        self.memory_list.memoryActivated.connect(self.apply_overlay_from_path)

        self.btn_toggle_view = QPushButton("Show Mixed")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.clicked.connect(self.toggle_view)

        self.btn_new_capture = QPushButton("New Capture")
        self.btn_new_capture.clicked.connect(self.reset_capture)

        self.btn_reset_selection = QPushButton("Reset Selection")
        self.btn_reset_selection.clicked.connect(self.reset_selection)

        self.btn_auto_quad = QPushButton("Auto-Suggest Quad")
        self.btn_auto_quad.clicked.connect(self.auto_suggest_quad)

        self.btn_import = QPushButton("Import Memories")
        self.btn_import.clicked.connect(self.import_memories)

        self.color_match_checkbox = QCheckBox("Color match overlays")
        self.color_match_checkbox.setChecked(True)
        self.color_match_checkbox.stateChanged.connect(self.toggle_color_match)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_change)
        self.opacity_label = QLabel("Opacity: 1.0")

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(50, 150)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.on_brightness_change)
        self.brightness_label = QLabel("Brightness: 1.0")

        self.btn_save_scene = QPushButton("Save Scene")
        self.btn_save_scene.clicked.connect(self.save_scene_file)
        self.btn_load_scene = QPushButton("Load Scene")
        self.btn_load_scene.clicked.connect(self.load_scene_file)

        self.overlay_list = QListWidget()
        self.overlay_list_label = QLabel("Overlays")
        self.overlay_list.currentRowChanged.connect(self.on_overlay_selected)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_toggle_view)
        controls.addWidget(self.btn_new_capture)
        controls.addWidget(self.btn_reset_selection)
        controls.addWidget(self.btn_auto_quad)
        controls.addWidget(self.btn_import)
        controls.addWidget(self.color_match_checkbox)
        controls.addWidget(self.btn_save_scene)
        controls.addWidget(self.btn_load_scene)
        controls.addStretch()
        controls.addWidget(self.status_label)

        left = QVBoxLayout()
        left.addWidget(self.camera_view, stretch=1)
        left.addLayout(controls)

        sliders = QHBoxLayout()
        sliders.addWidget(self.opacity_label)
        sliders.addWidget(self.opacity_slider)
        sliders.addWidget(self.brightness_label)
        sliders.addWidget(self.brightness_slider)
        left.addLayout(sliders)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left, stretch=3)
        right = QVBoxLayout()
        right.addWidget(QLabel("Memories"))
        right.addWidget(self.memory_list)
        right.addWidget(self.overlay_list_label)
        right.addWidget(self.overlay_list)
        main_layout.addLayout(right, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.camera_view.pointClicked.connect(self.on_point_clicked)
        self.camera_view.imageDropped.connect(self.apply_overlay_from_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 fps

    def closeEvent(self, event):
        self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

    def update_frame(self):
        if not self.paused:
            ret, frame = self.capture.read()
            if not ret:
                self.status_label.setText("Camera read failed.")
                return
            self.current_frame = frame
            # Try reapplying overlays when live
            self.try_reapply_overlays()
        self.refresh_view()

    def refresh_view(self):
        frame = None
        if self.show_mixed and self.mixed_frame is not None:
            frame = self.mixed_frame.copy()
        elif self.current_frame is not None:
            frame = self.current_frame.copy()

        if frame is None:
            return

        frame = draw_selection(frame, self.points)
        pixmap = bgr_to_qpixmap(frame)

        if self.camera_view.width() != pixmap.width() or self.camera_view.height() != pixmap.height():
            self.camera_view.setFixedSize(pixmap.width(), pixmap.height())

        self.camera_view.setPixmap(pixmap)

    def on_point_clicked(self, pos):
        if self.current_frame is None:
            return
        if len(self.points) >= 4:
            self.status_label.setText("Already have 4 points. Reset to change selection.")
            return
        x, y = pos
        frame_h, frame_w = self.current_frame.shape[:2]
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        self.points.append((x, y))
        if len(self.points) == 4:
            self.status_label.setText("Points locked. Drag/drop a memory image onto the view.")
        else:
            self.status_label.setText(f"Point {len(self.points)} set. Click {4 - len(self.points)} more.")
        self.refresh_view()

    def apply_overlay_from_path(self, path: str):
        path = _clean_path(path)
        if len(self.points) != 4:
            self.status_label.setText("Select 4 corners first.")
            return
        if self.current_frame is None:
            self.status_label.setText("No camera frame available.")
            return
        try:
            overlay_img = load_image(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load image: {exc}")
            return

        mixed = overlay_image(
            self.current_frame.copy(),
            overlay_img,
            self.points,
            match_colors=self.match_colors,
            opacity=self.default_opacity,
            brightness=self.default_brightness,
        )
        self.mixed_frame = mixed
        self.show_mixed = True
        self.btn_toggle_view.setChecked(True)
        self.btn_toggle_view.setText("Show Real")
        self.paused = True
        self.status_label.setText("Overlay placed. Toggle view or capture new.")
        # Persist overlay with scene features for reattachment
        base_kp, base_des = compute_features(self.current_frame)
        record = {
            "overlay_img": overlay_img,
            "image_path": path,
            "base_quad": list(self.points),
            "base_kp": base_kp,
            "base_des": base_des,
            "opacity": self.default_opacity,
            "brightness": self.default_brightness,
        }
        self.overlay_records.append(record)
        self.rebuild_overlay_list()
        self.refresh_view()

    def toggle_view(self):
        self.show_mixed = self.btn_toggle_view.isChecked()
        self.btn_toggle_view.setText("Show Real" if self.show_mixed else "Show Mixed")
        self.refresh_view()

    def reset_selection(self):
        self.points = []
        self.status_label.setText("Selection cleared. Click 4 corners.")
        self.refresh_view()

    def reset_capture(self):
        self.paused = False
        self.mixed_frame = None
        self.show_mixed = False
        self.btn_toggle_view.setChecked(False)
        self.btn_toggle_view.setText("Show Mixed")
        self.points = []
        self.overlay_records = []
        self.rebuild_overlay_list()
        self.status_label.setText("Live. Click 4 corners to place a memory.")
        self.refresh_view()

    def import_memories(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select memory images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not paths:
            return
        for path in paths:
            item = QListWidgetItem(Path(path).stem)
            pixmap = QPixmap(path).scaled(
                160, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            item.setIcon(QIcon(pixmap))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.memory_list.addItem(item)
        self.status_label.setText(f"Added {len(paths)} memories.")

    def auto_suggest_quad(self):
        if self.current_frame is None:
            self.status_label.setText("No camera frame to analyze.")
            return
        quad = suggest_quad(self.current_frame)
        if quad is None:
            self.status_label.setText("No suitable plane found.")
            return
        self.points = [(int(x), int(y)) for x, y in quad]
        self.status_label.setText("Auto-selected plane. Drag a memory onto the view.")
        self.refresh_view()

    def toggle_color_match(self, state):
        self.match_colors = bool(state)
        self.status_label.setText(f"Color match {'on' if self.match_colors else 'off'}.")

    def try_reapply_overlays(self):
        if not self.overlay_records:
            return
        frame = self.current_frame.copy()
        any_attached = False
        for record in self.overlay_records:
            quad = reproject_quad(record["base_kp"], record["base_des"], frame, record["base_quad"])
            if quad is None:
                continue
            frame = overlay_image(
                frame,
                record["overlay_img"],
                quad,
                match_colors=self.match_colors,
                opacity=record.get("opacity", 1.0),
                brightness=record.get("brightness", 1.0),
            )
            any_attached = True
        if any_attached:
            self.mixed_frame = frame
            # Keep view preference; do not force toggle

    def rebuild_overlay_list(self):
        self.overlay_list.clear()
        for idx, rec in enumerate(self.overlay_records, start=1):
            item = QListWidgetItem(f"{idx}: {Path(rec.get('image_path', 'overlay')).name}")
            self.overlay_list.addItem(item)

    def on_overlay_selected(self, row: int):
        if row < 0 or row >= len(self.overlay_records):
            return
        rec = self.overlay_records[row]
        self.opacity_slider.blockSignals(True)
        self.brightness_slider.blockSignals(True)
        self.opacity_slider.setValue(int(rec.get("opacity", 1.0) * 100))
        self.brightness_slider.setValue(int(rec.get("brightness", 1.0) * 100))
        self.opacity_slider.blockSignals(False)
        self.brightness_slider.blockSignals(False)

    def on_opacity_change(self, value):
        self.default_opacity = value / 100.0
        self.opacity_label.setText(f"Opacity: {self.default_opacity:.2f}")
        idx = self.overlay_list.currentRow()
        if 0 <= idx < len(self.overlay_records):
            self.overlay_records[idx]["opacity"] = self.default_opacity
            self.status_label.setText("Updated overlay opacity.")
            self.try_reapply_overlays()
            self.refresh_view()

    def on_brightness_change(self, value):
        self.default_brightness = value / 100.0
        self.brightness_label.setText(f"Brightness: {self.default_brightness:.2f}")
        idx = self.overlay_list.currentRow()
        if 0 <= idx < len(self.overlay_records):
            self.overlay_records[idx]["brightness"] = self.default_brightness
            self.status_label.setText("Updated overlay brightness.")
            self.try_reapply_overlays()
            self.refresh_view()

    def save_scene_file(self):
        if not self.overlay_records:
            self.status_label.setText("No overlays to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Scene", "scenes/scene.json", "Scene (*.json)")
        if not path:
            return
        scene_path = Path(path)
        save_scene(scene_path, self.overlay_records)
        self.status_label.setText(f"Saved scene to {scene_path}.")

    def load_scene_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Scene", "scenes", "Scene (*.json)")
        if not path:
            return
        scene_path = Path(path)
        try:
            loaded = load_scene(scene_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load scene: {exc}")
            return
        new_records = []
        for rec in loaded:
            try:
                overlay_img = load_image(rec["image_path"])
            except Exception:
                continue
            rec["overlay_img"] = overlay_img
            new_records.append(rec)
        self.overlay_records.extend(new_records)
        self.rebuild_overlay_list()
        self.status_label.setText(f"Loaded scene with {len(new_records)} overlays.")
        self.paused = False


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
