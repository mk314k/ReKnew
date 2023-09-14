import json
from pathlib import Path

import numpy as np
import pytest

from reknew.compositing import order_quad
from reknew.images import load_memory_image
from reknew.scene import Overlay, RoomFeatures, Scene, load_scene, save_scene


def make_scene(image_path: Path, room_frame) -> Scene:
    overlay = Overlay(
        image=load_memory_image(image_path),
        image_path=str(image_path),
        quad=order_quad([(100, 100), (300, 100), (300, 300), (100, 300)]),
        opacity=0.8,
        brightness=1.2,
    )
    return Scene([overlay], RoomFeatures.detect(room_frame))


def test_room_features_recognize_the_same_view(textured_room):
    homography = RoomFeatures.detect(textured_room).homography_to(textured_room)
    assert homography is not None
    assert np.allclose(homography, np.eye(3), atol=0.01)


def test_room_features_follow_a_camera_shift(textured_room):
    shifted = np.roll(textured_room, shift=(12, 20), axis=(0, 1))
    homography = RoomFeatures.detect(textured_room).homography_to(shifted)
    assert homography is not None
    assert np.allclose(homography[:2, 2], [20, 12], atol=1.0)


def test_room_features_reject_a_different_room(textured_room, room):
    assert RoomFeatures.detect(textured_room).homography_to(room) is None


def test_scene_round_trip(tmp_path, textured_room, memory, memory_path):
    scene_path = tmp_path / "scenes" / "room.json"
    original = make_scene(memory_path, textured_room)
    save_scene(scene_path, original)

    stored = json.loads(scene_path.read_text())["overlays"][0]["image_path"]
    loaded, skipped = load_scene(scene_path)

    assert skipped == 0
    assert Path(stored) == Path("..") / "memories" / "memory.png"
    overlay, saved = loaded.overlays[0], original.overlays[0]
    assert Path(overlay.image_path).resolve() == memory_path.resolve()
    assert np.allclose(overlay.quad, saved.quad)
    assert (overlay.opacity, overlay.brightness) == (0.8, 1.2)
    assert overlay.image.shape == memory.shape
    assert len(loaded.room_features.keypoints) == len(original.room_features.keypoints)
    assert np.array_equal(loaded.room_features.descriptors, original.room_features.descriptors)


def test_load_scene_skips_missing_images(tmp_path, textured_room, memory_path):
    scene_path = tmp_path / "room.json"
    scene = make_scene(memory_path, textured_room)
    scene.overlays[0].image_path = str(tmp_path / "missing.png")
    save_scene(scene_path, scene)
    loaded, skipped = load_scene(scene_path)
    assert loaded.overlays == [] and skipped == 1


def test_load_scene_rejects_other_versions(tmp_path, textured_room, memory_path):
    scene_path = tmp_path / "room.json"
    save_scene(scene_path, make_scene(memory_path, textured_room))
    data = json.loads(scene_path.read_text())
    data["version"] = 2
    scene_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported scene version"):
        load_scene(scene_path)
