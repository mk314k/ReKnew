import numpy as np

from reknew.projectors import SimulatedProjector
from reknew.session import ProjectionMode, ProjectionSession, ProjectionSettings


def make_session(room, **settings) -> ProjectionSession:
    session = ProjectionSession(SimulatedProjector(), ProjectionSettings(**settings))
    session.process_frame(room)
    return session


def test_idle_until_an_overlay_is_placed(room):
    session = make_session(room)
    assert session.stats.mode is ProjectionMode.IDLE
    assert not session.projector.image_lin.any()


def test_placing_an_overlay_lights_the_projector(room, memory, overlay_corners):
    session = make_session(room, room_light=0.3)
    session.add_overlay(memory, "memory.png", overlay_corners[::-1])
    session.process_frame(room)

    assert session.overlays[0].quad.tolist() == overlay_corners.tolist()
    assert session.stats.mode is ProjectionMode.COMPENSATED
    assert session.stats.contrast > 1.0
    assert session.projector.image_lin.any()
    assert session.render_view().shape == room.shape


def test_raw_mode_when_compensation_is_off(room, memory, overlay_corners):
    session = make_session(room, compensate=False)
    session.add_overlay(memory, "memory.png", overlay_corners)
    session.process_frame(room)
    assert session.stats.mode is ProjectionMode.RAW
    assert session.unreachable is None


def test_saved_scene_reanchors_after_camera_moves(textured_room, memory, overlay_corners):
    session = make_session(textured_room)
    session.add_overlay(memory, "memory.png", overlay_corners)
    scene = session.snapshot_scene()

    shifted = np.roll(textured_room, shift=(12, 20), axis=(0, 1))
    later = make_session(shifted)
    assert later.add_scene(scene)
    assert np.abs(later.overlays[0].quad - (overlay_corners + [20, 12])).max() < 2.0


def test_calibration_steps(room, memory, overlay_corners):
    session = make_session(room, room_light=0.3, simulated_gain=1.5)
    session.add_overlay(memory, "memory.png", overlay_corners)

    session.begin_calibration()
    session.process_frame(room)  # frames keep arriving while the caller waits
    session.project_calibration_pattern()
    session.process_frame(room)
    result = session.finish_calibration()

    assert result.succeeded
    assert result.footprint_error < 1.0
    assert abs(result.gain - 1.5) < 0.08
    assert session.is_calibrated and not session.is_busy
