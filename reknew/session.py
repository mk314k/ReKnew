"""Projector-camera session: overlays, calibration and per-frame rendering, independent of any UI."""

from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np

from reknew.calibration import (
    ChessboardPattern,
    detect_projector_homography,
    footprint_error,
    make_chessboard_pattern,
    measure_projector_gain,
)
from reknew.compositing import Layer, Size, composite_layers, order_quad
from reknew.light import estimate_reflectance, linear_to_srgb, srgb_to_linear
from reknew.projection import (
    compensation_target,
    footprint_corners,
    footprint_mask,
    simulate_projection,
    solve_projector_light,
)
from reknew.projectors import Projector, black_frame
from reknew.scene import Overlay, RoomFeatures, Scene


class ProjectionMode(Enum):
    IDLE = "idle"  # nothing to project
    RAW = "raw"  # memories projected as they are
    NEEDS_ROOM = "needs_room"  # compensation is on, but the room hasn't been captured yet
    COMPENSATED = "compensated"


@dataclass
class ProjectionSettings:
    compensate: bool = True
    uniformity: float = 0.5
    room_light: float = 1.0  # simulation only
    simulated_gain: float = 1.0  # simulation only


@dataclass(frozen=True)
class ProjectionStats:
    mode: ProjectionMode
    contrast: float | None = None
    unreachable_share: float | None = None


@dataclass(frozen=True)
class CalibrationResult:
    succeeded: bool
    gain: float | None = None
    footprint_error: float | None = None  # simulation only: px from the projector's true position


class ProjectionSession:
    """
    Feed camera frames in with `process_frame`; the session keeps the projector showing the
    current overlays. Calibration and room capture are two-step so the caller can wait for
    the projector and camera to settle between steps.
    """

    def __init__(self, projector: Projector, settings: ProjectionSettings | None = None):
        self.settings = settings or ProjectionSettings()
        self.overlays: list[Overlay] = []
        self.live_frame: np.ndarray | None = None
        self.live_lin: np.ndarray | None = None
        self.unreachable: np.ndarray | None = None
        self.stats = ProjectionStats(ProjectionMode.IDLE)
        self.projector: Projector | None = None
        self._composite_cache: dict[bool, tuple] = {}
        self._footprint_cache: tuple | None = None
        self.set_projector(projector)

    # ----- projector & geometry -----------------------------------------------------------

    def set_projector(self, projector: Projector) -> None:
        if self.projector is not None:
            self.projector.close()
        self.projector = projector
        self.room_lin: np.ndarray | None = None  # real projector: room captured with the projector dark
        self.measured_gain: float | None = None
        self.is_calibrated = False
        self.is_busy = False
        self._calibration_ambient: np.ndarray | None = None
        self._calibration_pattern: ChessboardPattern | None = None
        self._set_geometry(None)

    def _set_geometry(self, proj_to_cam: np.ndarray | None) -> None:
        self.proj_to_cam = proj_to_cam
        self.cam_to_proj = None if proj_to_cam is None else np.linalg.inv(proj_to_cam)

    @property
    def cam_size(self) -> Size:
        h, w = self.live_frame.shape[:2]
        return (w, h)

    @property
    def gain(self) -> float:
        """Projector brightness used for compensation: measured if calibrated, else the best guess."""
        if self.measured_gain is not None:
            return self.measured_gain
        return self.settings.simulated_gain if self.projector.is_simulated else 1.0

    def footprint(self) -> np.ndarray | None:
        if self.proj_to_cam is None:
            return None
        return footprint_corners(self.proj_to_cam, self.projector.size)

    # ----- frames -------------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> None:
        self.live_frame = frame
        self.live_lin = srgb_to_linear(frame)
        if self.projector.is_simulated and not self.projector.is_mounted:
            self.projector.mount_next_to_camera(self.cam_size)
            self._set_geometry(self.projector.true_proj_to_cam)
        if not self.is_busy:
            self.projector.show(self._render_projector_image())

    def room_frame(self) -> np.ndarray:
        """The room as seen with the projector dark, uint8 BGR."""
        if not self.projector.is_simulated and self.room_lin is not None:
            return linear_to_srgb(self.room_lin)
        return self.live_frame

    def ambient_lin(self) -> np.ndarray | None:
        if self.projector.is_simulated:
            return self.live_lin * self.settings.room_light
        return self.room_lin

    def capture_lin(self) -> np.ndarray:
        """What the camera sees right now, including projector light."""
        if self.projector.is_simulated:
            return self.projector.capture(self.live_lin, self.settings.room_light, self.settings.simulated_gain)
        return self.live_lin

    def render_view(self) -> np.ndarray:
        """How the room looks with the projection, uint8 BGR (simulated, or predicted for a real projector)."""
        if self.projector.is_simulated:
            return linear_to_srgb(self.capture_lin())
        if self.room_lin is None or self.proj_to_cam is None:
            return self.live_frame.copy()
        return linear_to_srgb(
            simulate_projection(self.room_lin, self.projector.image_lin, self.proj_to_cam, self.gain)
        )

    def _render_projector_image(self) -> np.ndarray:
        self.unreachable = None
        if not self.overlays or self.proj_to_cam is None:
            self.stats = ProjectionStats(ProjectionMode.IDLE)
            return black_frame(self.projector.size)

        ambient = self.ambient_lin()
        if not self.settings.compensate or ambient is None:
            mode = ProjectionMode.NEEDS_ROOM if self.settings.compensate else ProjectionMode.RAW
            self.stats = ProjectionStats(mode)
            content, _ = self._overlay_composite(in_projector_space=True)
            return np.clip(content, 0.0, 1.0)

        reflectance = estimate_reflectance(ambient)
        content, coverage = self._overlay_composite(in_projector_space=False)
        target, contrast = compensation_target(
            content, coverage, ambient, reflectance, self.gain, self.settings.uniformity
        )
        light, unreachable = solve_projector_light(target, ambient, reflectance, self.gain)
        overlay_area = coverage > 0
        unreachable |= overlay_area & ~self._footprint_mask()
        self.unreachable = unreachable
        self.stats = ProjectionStats(
            ProjectionMode.COMPENSATED,
            contrast=contrast.ratio if contrast else None,
            unreachable_share=float(unreachable.sum()) / max(int(overlay_area.sum()), 1),
        )
        return cv2.warpPerspective(light, self.cam_to_proj, self.projector.size)

    def _overlay_composite(self, in_projector_space: bool) -> tuple[np.ndarray, np.ndarray]:
        """Overlays composited over black in camera or projector space, cached until they change."""
        size = self.projector.size if in_projector_space else self.cam_size
        key = (
            size,
            self.proj_to_cam.tobytes(),
            tuple((id(o.image), o.quad.tobytes(), o.opacity, o.brightness) for o in self.overlays),
        )
        cached = self._composite_cache.get(in_projector_space)
        if cached is not None and cached[0] == key:
            return cached[1]

        layers = []
        for overlay in self.overlays:
            quad = overlay.quad
            if in_projector_space:
                quad = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), self.cam_to_proj).reshape(4, 2)
            layers.append(Layer(overlay.image, quad, overlay.opacity, overlay.brightness))
        result = composite_layers(layers, size)
        self._composite_cache[in_projector_space] = (key, result)
        return result

    def _footprint_mask(self) -> np.ndarray:
        key = (self.proj_to_cam.tobytes(), self.projector.size, self.cam_size)
        if self._footprint_cache is None or self._footprint_cache[0] != key:
            self._footprint_cache = (key, footprint_mask(self.proj_to_cam, self.projector.size, self.cam_size))
        return self._footprint_cache[1]

    # ----- overlays -----------------------------------------------------------------------

    def add_overlay(self, image: np.ndarray, image_path: str, corners,
                    opacity: float = 1.0, brightness: float = 1.0) -> Overlay:
        overlay = Overlay(image, image_path, order_quad(corners), opacity=opacity, brightness=brightness)
        self.overlays.append(overlay)
        return overlay

    def snapshot_scene(self) -> Scene:
        overlays = [replace(overlay, quad=overlay.quad.copy()) for overlay in self.overlays]
        return Scene(overlays, RoomFeatures.detect(self.room_frame()))

    def add_scene(self, scene: Scene) -> bool:
        """
        Add a saved scene's overlays, moved to where the saved view appears in the current room.
        Returns False if the room wasn't recognized; the overlays then keep their saved positions.
        """
        homography = scene.room_features.homography_to(self.room_frame())
        if homography is not None:
            for overlay in scene.overlays:
                moved = cv2.perspectiveTransform(overlay.quad.reshape(-1, 1, 2), homography)
                overlay.quad = order_quad(moved)
        self.overlays.extend(scene.overlays)
        return homography is not None

    def clear_overlays(self) -> None:
        self.overlays = []

    # ----- room capture & calibration -----------------------------------------------------

    def begin_room_capture(self) -> None:
        self.is_busy = True
        self.projector.show(black_frame(self.projector.size))

    def finish_room_capture(self) -> None:
        self.room_lin = self.capture_lin()
        self.is_busy = False

    def begin_calibration(self) -> None:
        self.is_busy = True
        self.unreachable = None
        self.projector.show(black_frame(self.projector.size))

    def project_calibration_pattern(self) -> None:
        self._calibration_ambient = self.capture_lin()
        self._calibration_pattern = make_chessboard_pattern(self.projector.size)
        self.projector.show(self._calibration_pattern.image_lin)

    def finish_calibration(self) -> CalibrationResult:
        ambient, pattern = self._calibration_ambient, self._calibration_pattern
        observed = self.capture_lin()
        self.is_busy = False
        self._calibration_ambient = self._calibration_pattern = None

        proj_to_cam = detect_projector_homography(ambient, observed, pattern)
        if proj_to_cam is None:
            return CalibrationResult(succeeded=False)

        self._set_geometry(proj_to_cam)
        self.is_calibrated = True
        self.measured_gain = measure_projector_gain(ambient, observed, pattern, proj_to_cam)
        error = None
        if self.projector.is_simulated:
            error = footprint_error(proj_to_cam, self.projector.true_proj_to_cam, self.projector.size)
        else:
            self.room_lin = ambient
        return CalibrationResult(succeeded=True, gain=self.measured_gain, footprint_error=error)
