# modules/image_overlay.py

import cv2
import numpy as np


def color_match(overlay_rgb, target_rgb, mask):
    """Simple mean/std matching inside masked region."""
    mask_f = mask.astype(np.float32) / 255.0
    mask_3 = np.dstack([mask_f] * 3)
    eps = 1e-3

    def stats(img):
        pixels = img * mask_3
        count = mask_f.sum()
        if count < 1:
            return 0, 1
        mean = pixels.sum(axis=(0, 1)) / (count + eps)
        std = np.sqrt(((pixels - mean) ** 2).sum(axis=(0, 1)) / (count + eps))
        return mean, np.maximum(std, eps)

    o_mean, o_std = stats(overlay_rgb.astype(np.float32))
    t_mean, t_std = stats(target_rgb.astype(np.float32))

    adjusted = (overlay_rgb.astype(np.float32) - o_mean) / o_std * t_std + t_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def overlay_image(background, overlay_img, destination_points, match_colors=True, opacity=1.0, brightness=1.0):
    """
    Perspective-warp overlay_img onto background using four destination points.
    destination_points: list of four (x, y) tuples in display order around the surface.
    """
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay_img.shape[:2]

    # Guarantee an alpha channel for blending
    if overlay_img.shape[2] == 3:
        alpha_channel = np.ones((h_ov, w_ov), dtype=np.uint8) * 255
        overlay_rgba = np.dstack([overlay_img, alpha_channel])
    else:
        overlay_rgba = overlay_img

    src_pts = np.array(
        [[0, 0], [w_ov, 0], [w_ov, h_ov], [0, h_ov]], dtype=np.float32
    )
    dst_pts = np.array(destination_points, dtype=np.float32)

    # Compute warp and apply to the overlay and alpha mask
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_overlay = cv2.warpPerspective(overlay_rgba[:, :, :3], matrix, (w_bg, h_bg))
    warped_alpha_raw = cv2.warpPerspective(
        overlay_rgba[:, :, 3], matrix, (w_bg, h_bg)
    )
    warped_alpha = warped_alpha_raw.astype(np.float32) / 255.0

    # Adjust brightness before blending
    warped_overlay = np.clip(warped_overlay.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    # Optional color match to ambient scene where the overlay lands
    if match_colors:
        mask = warped_alpha_raw
        roi_bg = background.copy()
        warped_overlay = color_match(warped_overlay, roi_bg, mask)

    # Blend warped overlay into the background
    composite = background.astype(np.float32)
    for c in range(3):
        composite[:, :, c] = (
            composite[:, :, c] * (1 - warped_alpha * opacity)
            + warped_overlay[:, :, c] * (warped_alpha * opacity)
        )

    return composite.clip(0, 255).astype(np.uint8)
