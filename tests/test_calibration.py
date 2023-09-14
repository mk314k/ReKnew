import pytest

from reknew.calibration import (
    detect_projector_homography,
    footprint_error,
    make_chessboard_pattern,
    measure_projector_gain,
)
from reknew.light import srgb_to_linear
from reknew.projectors import SimulatedProjector, black_frame
from tests.scenery import CAMERA_SIZE


@pytest.mark.parametrize("room_light, gain", [(1.0, 1.0), (0.3, 1.0), (0.3, 2.0), (0.1, 0.5)])
def test_calibration_recovers_simulated_projector(room, room_light, gain):
    room_lin = srgb_to_linear(room)
    projector = SimulatedProjector()
    projector.mount_next_to_camera(CAMERA_SIZE)

    projector.show(black_frame(projector.size))
    ambient = projector.capture(room_lin, room_light, gain)
    pattern = make_chessboard_pattern(projector.size)
    projector.show(pattern.image_lin)
    observed = projector.capture(room_lin, room_light, gain)

    proj_to_cam = detect_projector_homography(ambient, observed, pattern)
    assert proj_to_cam is not None
    assert footprint_error(proj_to_cam, projector.true_proj_to_cam, projector.size) < 1.0
    assert measure_projector_gain(ambient, observed, pattern, proj_to_cam) == pytest.approx(gain, rel=0.05)


def test_calibration_fails_cleanly_without_a_pattern(room):
    room_lin = srgb_to_linear(room)
    pattern = make_chessboard_pattern((1280, 720))
    assert detect_projector_homography(room_lin, room_lin, pattern) is None
