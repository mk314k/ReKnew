import numpy as np

from reknew.demo import DEMO_CAMERA_SIZE, DemoRoomCamera, ensure_sample_memories
from reknew.surface_detection import suggest_quad


def test_demo_room_camera_streams_valid_frames_with_detectable_quad():
    cam = DemoRoomCamera()
    assert cam.isOpened()
    ok, frame = cam.read()
    assert ok
    assert frame.shape == (DEMO_CAMERA_SIZE[1], DEMO_CAMERA_SIZE[0], 3)
    assert frame.dtype == np.uint8

    quad = suggest_quad(frame)
    assert quad is not None
    assert quad.shape == (4, 2)

    cam.release()
    assert not cam.isOpened()


def test_ensure_sample_memories_populates_empty_dir(tmp_path):
    mem_dir = tmp_path / "memories"
    created = ensure_sample_memories(mem_dir)
    assert len(created) == 3
    assert all(p.exists() for p in created)

    # Calling again does not overwrite or duplicate existing images
    second = ensure_sample_memories(mem_dir)
    assert len(second) == 3
