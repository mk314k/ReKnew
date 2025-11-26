import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def compute_features(frame: np.ndarray, max_features: int = 600):
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors


def reproject_quad(base_keypoints, base_descriptors, current_frame, quad, min_matches=12):
    """
    Attempt to find homography from base features to current frame and transform quad.
    Returns transformed quad (4x2 float) or None.
    """
    if base_descriptors is None or len(base_keypoints) < min_matches:
        return None

    current_kp, current_des = compute_features(current_frame)
    if current_des is None or len(current_kp) < min_matches:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(base_descriptors, current_des, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None

    src_pts = np.float32([base_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    quad_pts = np.float32(quad).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(quad_pts, H)
    return transformed.reshape(4, 2)


def _serialize_kp(kp):
    return {"pt": kp.pt}


def _deserialize_kp(kp_dict):
    x, y = kp_dict["pt"]
    return cv2.KeyPoint(x, y, 1)


def save_scene(scene_path: Path, overlay_records: List[Dict[str, Any]]):
    scene = {"overlays": []}
    for rec in overlay_records:
        scene["overlays"].append(
            {
                "image_path": rec["image_path"],
                "base_quad": rec["base_quad"],
                "opacity": rec.get("opacity", 1.0),
                "brightness": rec.get("brightness", 1.0),
                "base_kp": [_serialize_kp(kp) for kp in rec["base_kp"]],
                "base_des": rec["base_des"].tolist() if rec["base_des"] is not None else None,
            }
        )
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(json.dumps(scene, indent=2))


def load_scene(scene_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(scene_path.read_text())
    overlays = []
    for rec in data.get("overlays", []):
        kp = [_deserialize_kp(k) for k in rec.get("base_kp", [])]
        des_list = rec.get("base_des")
        des = np.array(des_list, dtype=np.uint8) if des_list is not None else None
        overlays.append(
            {
                "image_path": rec["image_path"],
                "base_quad": rec["base_quad"],
                "opacity": rec.get("opacity", 1.0),
                "brightness": rec.get("brightness", 1.0),
                "base_kp": kp,
                "base_des": des,
            }
        )
    return overlays
