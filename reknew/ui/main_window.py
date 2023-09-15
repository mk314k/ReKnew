import sys
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from reknew.demo import DemoRoomCamera, ensure_sample_memories
from reknew.images import load_memory_image
from reknew.light import linear_to_srgb
from reknew.paths import MEMORIES_DIR, SCENES_DIR
from reknew.projectors import SimulatedProjector
from reknew.scene import load_scene, save_scene
from reknew.session import CalibrationResult, ProjectionMode, ProjectionSession, ProjectionStats
from reknew.surface_detection import suggest_quad
from reknew.ui.annotations import draw_corners, draw_outline, tint
from reknew.ui.screen_projector import ScreenProjector
from reknew.ui.widgets import CameraView, LabeledSlider, MemoryList, bgr_to_pixmap

FRAME_INTERVAL_MS = 30
SIMULATED_SETTLE_MS = 100
REAL_SETTLE_MS = 800  # let the projector update and the camera catch up before capturing
PREVIEW_SIZE = (320, 180)
SIMULATED_MODE, REAL_MODE = 0, 1


def describe_stats(stats: ProjectionStats) -> str:
    if stats.mode is ProjectionMode.RAW:
        return "Projecting raw memories (no compensation)."
    if stats.mode is ProjectionMode.NEEDS_ROOM:
        return "Capture Room to enable compensation."
    if stats.mode is ProjectionMode.COMPENSATED:
        contrast = "n/a" if stats.contrast is None else f"{stats.contrast:.1f}:1"
        return f"Contrast {contrast}, unreachable {stats.unreachable_share:.0%} of overlay area"
    return ""


def describe_calibration(result: CalibrationResult, true_gain: float | None) -> str:
    if not result.succeeded:
        return "Calibration failed: chessboard not found. Aim at a light, flat area."
    gain = "n/a" if result.gain is None else f"{result.gain:.2f}"
    if result.footprint_error is None:
        return f"Calibrated. Estimated projector gain {gain}."
    return f"Calibrated. Corner error {result.footprint_error:.1f}px, gain {gain} (true {true_gain:.2f})."


class MainWindow(QMainWindow):
    def __init__(self, camera_index: int = 0, demo: bool = False):
        super().__init__()
        self.setWindowTitle("ReKnew - Projection Memory Overlay")

        ensure_sample_memories(MEMORIES_DIR)
        if demo:
            self.camera = DemoRoomCamera()
        else:
            cam = cv2.VideoCapture(camera_index)
            if cam.isOpened():
                self.camera = cam
            else:
                cam.release()
                self.camera = DemoRoomCamera()

        self.session = ProjectionSession(SimulatedProjector())
        self.corners: list[tuple[int, int]] = []
        self.show_camera = False
        self.highlight_unreachable = True

        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._update_mode_controls()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(FRAME_INTERVAL_MS)

    # ----- construction -------------------------------------------------------------------

    def _build_widgets(self):
        settings = self.session.settings
        self.camera_view = CameraView()
        self.status_label = QLabel("Click 4 corners of a surface.")
        self.stats_label = QLabel()
        self.memory_list = MemoryList(MEMORIES_DIR)
        self.overlay_list = QListWidget()
        self.projector_preview = QLabel()
        self.projector_preview.setFixedSize(*PREVIEW_SIZE)
        self.projector_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.projector_preview.setStyleSheet("background-color: #000;")

        self.view_button = QPushButton("Show Camera")
        self.view_button.setCheckable(True)
        self.reset_corners_button = QPushButton("Reset Selection")
        self.suggest_button = QPushButton("Auto-Suggest Quad")
        self.import_button = QPushButton("Import Memories")
        self.clear_button = QPushButton("Clear Overlays")
        self.save_button = QPushButton("Save Scene")
        self.load_button = QPushButton("Load Scene")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simulated projector", "Real projector"])
        self.calibrate_button = QPushButton("Calibrate")
        self.capture_room_button = QPushButton("Capture Room")
        self.capture_room_button.setToolTip("Capture the room with the projector dark.")
        self.compensate_checkbox = QCheckBox("Compensate for surface")
        self.compensate_checkbox.setChecked(settings.compensate)
        self.unreachable_checkbox = QCheckBox("Highlight unreachable")
        self.unreachable_checkbox.setChecked(self.highlight_unreachable)

        self.room_light_slider = LabeledSlider("Room light", 0.05, 1.0, settings.room_light)
        self.gain_slider = LabeledSlider("Projector brightness", 0.1, 3.0, settings.simulated_gain)
        self.uniformity_slider = LabeledSlider(
            "Uniformity", 0.0, 1.0, settings.uniformity,
            tooltip="High: hide surface colors evenly but lose contrast. Low: more contrast.",
        )
        self.opacity_slider = LabeledSlider("Opacity", 0.1, 1.0, 1.0)
        self.brightness_slider = LabeledSlider("Brightness", 0.5, 1.5, 1.0)

    def _build_layout(self):
        def row(*widgets, trailing=None):
            layout = QHBoxLayout()
            for widget in widgets:
                layout.addWidget(widget)
            if trailing is not None:
                layout.addStretch()
                layout.addWidget(trailing)
            return layout

        left = QVBoxLayout()
        left.addWidget(self.camera_view, stretch=1)
        left.addLayout(row(
            self.view_button, self.reset_corners_button, self.suggest_button, self.import_button,
            self.clear_button, self.save_button, self.load_button,
            trailing=self.status_label,
        ))
        left.addLayout(row(
            self.mode_combo, self.calibrate_button, self.capture_room_button,
            self.compensate_checkbox, self.unreachable_checkbox,
            trailing=self.stats_label,
        ))
        left.addLayout(row(
            self.room_light_slider, self.gain_slider, self.uniformity_slider,
            self.opacity_slider, self.brightness_slider,
        ))

        right = QVBoxLayout()
        for title, widget in (
            ("Memories", self.memory_list),
            ("Overlays", self.overlay_list),
            ("Projector output", self.projector_preview),
        ):
            right.addWidget(QLabel(title))
            right.addWidget(widget)

        main = QHBoxLayout()
        main.addLayout(left, stretch=3)
        main.addLayout(right, stretch=1)
        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

    def _connect_signals(self):
        settings = self.session.settings
        self.camera_view.clicked.connect(self._add_corner)
        self.camera_view.imageDropped.connect(self._place_memory)
        self.memory_list.memoryActivated.connect(self._place_memory)
        self.overlay_list.currentRowChanged.connect(self._select_overlay)

        self.view_button.toggled.connect(self._toggle_view)
        self.reset_corners_button.clicked.connect(self._reset_corners)
        self.suggest_button.clicked.connect(self._suggest_corners)
        self.import_button.clicked.connect(self._import_memories)
        self.clear_button.clicked.connect(self._clear_overlays)
        self.save_button.clicked.connect(self._save_scene)
        self.load_button.clicked.connect(self._load_scene)

        self.mode_combo.currentIndexChanged.connect(self._switch_projector)
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.capture_room_button.clicked.connect(self._start_room_capture)
        self.compensate_checkbox.toggled.connect(lambda on: setattr(settings, "compensate", on))
        self.unreachable_checkbox.toggled.connect(lambda on: setattr(self, "highlight_unreachable", on))

        self.room_light_slider.valueChanged.connect(lambda v: setattr(settings, "room_light", v))
        self.gain_slider.valueChanged.connect(self._set_simulated_gain)
        self.uniformity_slider.valueChanged.connect(lambda v: setattr(settings, "uniformity", v))
        self.opacity_slider.valueChanged.connect(lambda v: self._edit_selected_overlay(opacity=v))
        self.brightness_slider.valueChanged.connect(lambda v: self._edit_selected_overlay(brightness=v))

    def closeEvent(self, event):
        self.timer.stop()
        self.session.projector.close()
        if self.camera.isOpened():
            self.camera.release()
        super().closeEvent(event)

    # ----- frame loop ---------------------------------------------------------------------

    def _on_tick(self):
        ok, frame = self.camera.read()
        if not ok:
            self._set_status("Camera read failed.")
            return
        self.session.process_frame(frame)
        self._refresh_view()

    def _refresh_view(self):
        session = self.session
        if session.live_frame is None:
            return
        if self.show_camera:
            frame = session.live_frame.copy()
        else:
            frame = session.render_view()
            if self.highlight_unreachable and session.unreachable is not None:
                tint(frame, session.unreachable)
        footprint = session.footprint()
        if footprint is not None:
            draw_outline(frame, footprint)
        draw_corners(frame, self.corners)
        self.camera_view.show_frame(frame)
        self.stats_label.setText(describe_stats(session.stats))
        self._refresh_projector_preview()

    def _refresh_projector_preview(self):
        image_lin = self.session.projector.image_lin
        h, w = image_lin.shape[:2]
        scale = min(PREVIEW_SIZE[0] / w, PREVIEW_SIZE[1] / h)
        small = cv2.resize(image_lin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.projector_preview.setPixmap(bgr_to_pixmap(linear_to_srgb(small)))

    def _set_status(self, text: str):
        self.status_label.setText(text)

    # ----- corners & overlays -------------------------------------------------------------

    def _add_corner(self, x: int, y: int):
        frame = self.session.live_frame
        if frame is None:
            return
        if len(self.corners) >= 4:
            self._set_status("Already have 4 points. Reset to change selection.")
            return
        h, w = frame.shape[:2]
        self.corners.append((min(max(x, 0), w - 1), min(max(y, 0), h - 1)))
        remaining = 4 - len(self.corners)
        if remaining:
            self._set_status(f"Point {len(self.corners)} set. Click {remaining} more.")
        else:
            self._set_status("Points locked. Drag/drop a memory image onto the view.")

    def _reset_corners(self):
        self.corners = []
        self._set_status("Selection cleared. Click 4 corners.")

    def _suggest_corners(self):
        if self.session.live_frame is None:
            return
        quad = suggest_quad(self.session.room_frame())
        if quad is None:
            self._set_status("No suitable plane found.")
            return
        self.corners = [(int(x), int(y)) for x, y in quad]
        self._set_status("Auto-selected plane. Drag a memory onto the view.")

    def _place_memory(self, path: str):
        if len(self.corners) != 4:
            self._set_status("Select 4 corners first.")
            return
        if self.session.live_frame is None:
            self._set_status("No camera frame available.")
            return
        try:
            image = load_memory_image(path)
        except OSError as exc:
            self._set_status(f"Failed to load image: {exc}")
            return
        self.session.add_overlay(
            image, path, self.corners,
            opacity=self.opacity_slider.value(), brightness=self.brightness_slider.value(),
        )
        self.corners = []
        self._refresh_overlay_list()
        self.overlay_list.setCurrentRow(len(self.session.overlays) - 1)
        self._set_status("Overlay placed. Click 4 new corners for the next one.")

    def _clear_overlays(self):
        self.corners = []
        self.session.clear_overlays()
        self._refresh_overlay_list()
        self._set_status("Overlays cleared. Click 4 corners to place a memory.")

    def _import_memories(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select memory images", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        for path in paths:
            self.memory_list.add_memory(path)
        if paths:
            self._set_status(f"Added {len(paths)} memories.")

    def _refresh_overlay_list(self):
        self.overlay_list.clear()
        for index, overlay in enumerate(self.session.overlays, start=1):
            self.overlay_list.addItem(f"{index}: {Path(overlay.image_path).name}")

    def _selected_overlay(self):
        row = self.overlay_list.currentRow()
        overlays = self.session.overlays
        return overlays[row] if 0 <= row < len(overlays) else None

    def _select_overlay(self, _row: int):
        overlay = self._selected_overlay()
        if overlay is not None:
            self.opacity_slider.set_value(overlay.opacity)
            self.brightness_slider.set_value(overlay.brightness)

    def _edit_selected_overlay(self, **changes):
        overlay = self._selected_overlay()
        if overlay is None:
            return
        for name, value in changes.items():
            setattr(overlay, name, value)

    def _toggle_view(self, show_camera: bool):
        self.show_camera = show_camera
        self.view_button.setText("Show Projection" if show_camera else "Show Camera")

    # ----- projector mode, calibration & room capture -------------------------------------

    def _switch_projector(self, mode: int):
        if mode == REAL_MODE:
            screens = QGuiApplication.screens()
            if len(screens) < 2:
                self._set_status("Connect the projector as a second display first.")
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(SIMULATED_MODE)
                self.mode_combo.blockSignals(False)
                return
            self.session.set_projector(ScreenProjector(screens[-1]))
            self._set_status("Point the projector at the scene, then Calibrate.")
        else:
            self.session.set_projector(SimulatedProjector())
            self._set_status("Simulated projector (using its true position until you Calibrate).")
        self._update_mode_controls()

    def _update_mode_controls(self):
        simulated = self.session.projector.is_simulated
        self.room_light_slider.setEnabled(simulated)
        self.gain_slider.setEnabled(simulated)
        self.capture_room_button.setEnabled(not simulated)

    def _set_simulated_gain(self, gain: float):
        self.session.settings.simulated_gain = gain
        if self.session.measured_gain is not None:
            self._set_status("Projector brightness changed; Calibrate again to re-measure it.")

    def _settle_ms(self) -> int:
        return SIMULATED_SETTLE_MS if self.session.projector.is_simulated else REAL_SETTLE_MS

    def _can_start_capture(self) -> bool:
        return self.session.live_frame is not None and not self.session.is_busy

    def _start_room_capture(self):
        if not self._can_start_capture():
            return
        self.session.begin_room_capture()
        self._set_status("Capturing room...")
        QTimer.singleShot(self._settle_ms(), self._finish_room_capture)

    def _finish_room_capture(self):
        self.session.finish_room_capture()
        self._set_status("Room captured.")

    def _start_calibration(self):
        if not self._can_start_capture():
            return
        self.session.begin_calibration()
        self._set_status("Calibrating: capturing room...")
        QTimer.singleShot(self._settle_ms(), self._project_calibration_pattern)

    def _project_calibration_pattern(self):
        self.session.project_calibration_pattern()
        self._set_status("Calibrating: projecting chessboard...")
        QTimer.singleShot(self._settle_ms(), self._finish_calibration)

    def _finish_calibration(self):
        result = self.session.finish_calibration()
        self._set_status(describe_calibration(result, self.session.settings.simulated_gain))

    # ----- scenes -------------------------------------------------------------------------

    def _save_scene(self):
        if not self.session.overlays:
            self._set_status("No overlays to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Scene", str(SCENES_DIR / "scene.json"), "Scene (*.json)")
        if path:
            save_scene(Path(path), self.session.snapshot_scene())
            self._set_status(f"Saved scene to {path}.")

    def _load_scene(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Scene", str(SCENES_DIR), "Scene (*.json)")
        if not path or self.session.live_frame is None:
            return
        try:
            scene, skipped = load_scene(Path(path))
        except (OSError, ValueError, KeyError) as exc:
            self._set_status(f"Failed to load scene: {exc}")
            return
        recognized = self.session.add_scene(scene)
        self._refresh_overlay_list()
        placement = "re-anchored to the current room" if recognized else "room not recognized, kept saved positions"
        message = f"Loaded {len(scene.overlays)} overlays ({placement})."
        if skipped:
            message += f" {skipped} skipped (image missing)."
        self._set_status(message)


def run():
    import argparse

    parser = argparse.ArgumentParser(description="ReKnew - Projection Memory Overlay")
    parser.add_argument("--demo", action="store_true", help="Use built-in synthetic demo room instead of a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    window = MainWindow(camera_index=args.camera, demo=args.demo)
    window.show()
    sys.exit(app.exec())
