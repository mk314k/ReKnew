"""Overlays placed in the room, recognizing the room again with ORB features, and scene files."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from reknew.compositing import order_quad
from reknew.images import load_memory_image

SCENE_VERSION = 3


@dataclass
class Overlay:
    """A memory image pinned to a quad of the room, in camera pixels."""

    image: np.ndarray
    image_path: str
    quad: np.ndarray  # 4x2, clockwise from top-left
    opacity: float = 1.0
    brightness: float = 1.0


@dataclass
class RoomFeatures:
    """Distinctive points of one camera view of the room, used to find that view again."""

    keypoints: list
    descriptors: np.ndarray | None

    @classmethod
    def detect(cls, frame: np.ndarray, max_features: int = 600) -> "RoomFeatures":
        keypoints, descriptors = cv2.ORB_create(max_features).detectAndCompute(frame, None)
        return cls(list(keypoints), descriptors)

    def homography_to(self, frame: np.ndarray, min_matches: int = 12) -> np.ndarray | None:
        """Mapping from this view's pixels to `frame`'s pixels, or None if the room isn't recognized."""
        current = RoomFeatures.detect(frame)
        if (self.descriptors is None or current.descriptors is None
                or min(len(self.keypoints), len(current.keypoints)) < min_matches):
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        good = [
            pair[0]
            for pair in matcher.knnMatch(self.descriptors, current.descriptors, k=2)
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(good) < min_matches:
            return None

        source = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([current.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        return homography


@dataclass
class Scene:
    overlays: list[Overlay]
    room_features: RoomFeatures  # the view the overlay quads were placed in


def _path_for_scene(image_path: str, scene_dir: Path) -> str:
    """Relative path when the image and scene share a parent folder, otherwise absolute."""
    target, scene_dir = Path(image_path).resolve(), scene_dir.resolve()
    try:
        common = Path(os.path.commonpath([target, scene_dir]))
    except ValueError:
        return str(target)
    if common == Path(common.anchor):
        return str(target)
    return os.path.relpath(target, scene_dir)


def save_scene(scene_path: Path, scene: Scene) -> None:
    features = scene.room_features
    data = {
        "version": SCENE_VERSION,
        "room_features": {
            "points": [kp.pt for kp in features.keypoints],
            "descriptors": None if features.descriptors is None else features.descriptors.tolist(),
        },
        "overlays": [
            {
                "image_path": _path_for_scene(overlay.image_path, scene_path.parent),
                "quad": np.asarray(overlay.quad).tolist(),
                "opacity": overlay.opacity,
                "brightness": overlay.brightness,
            }
            for overlay in scene.overlays
        ],
    }
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(json.dumps(data, indent=2))


def load_scene(scene_path: Path) -> tuple[Scene, int]:
    """Load a scene and its images; returns (scene, number of overlays skipped because the image is missing)."""
    data = json.loads(scene_path.read_text())
    if data.get("version") != SCENE_VERSION:
        raise ValueError(f"unsupported scene version {data.get('version')!r}")

    features = data["room_features"]
    descriptors = features["descriptors"]
    room_features = RoomFeatures(
        keypoints=[cv2.KeyPoint(x, y, 1) for x, y in features["points"]],
        descriptors=None if descriptors is None else np.array(descriptors, dtype=np.uint8),
    )

    overlays, skipped = [], 0
    for entry in data["overlays"]:
        image_path = Path(entry["image_path"])
        if not image_path.is_absolute():
            image_path = scene_path.parent / image_path
        try:
            image = load_memory_image(image_path)
        except OSError:
            skipped += 1
            continue
        overlays.append(Overlay(
            image=image,
            image_path=str(image_path),
            quad=order_quad(entry["quad"]),
            opacity=entry["opacity"],
            brightness=entry["brightness"],
        ))
    return Scene(overlays, room_features), skipped
