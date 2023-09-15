"""
Built-in synthetic demo room camera and sample memory generator for zero-friction demos.

When no hardware webcam is available (or when `reknew --demo` is passed), `DemoRoomCamera`
provides a realistic 640x480 room view (warm plaster wall, wood wainscoting, a framed dark-red
poster that `Auto-Suggest Quad` detects immediately, and subtle sensor noise so ORB features
and projector calibration work identically to a physical camera).
"""

from pathlib import Path

import cv2
import numpy as np

from reknew.compositing import Size

DEMO_CAMERA_SIZE: Size = (640, 480)


def make_demo_room_base(size: Size = DEMO_CAMERA_SIZE) -> np.ndarray:
    """Create a realistic BGR room frame with textured plaster wall, wood trim, and a framed poster."""
    w, h = size
    rng = np.random.default_rng(42)

    # 1. Warm plaster wall with vertical lighting gradient
    y_grad = np.linspace(1.04, 0.88, h, dtype=np.float32)[:, None, None]
    base_wall = np.array([175, 192, 206], dtype=np.float32) * y_grad
    room = np.clip(base_wall, 0, 255).astype(np.uint8)
    room = np.broadcast_to(room, (h, w, 3)).copy()

    # Subtle plaster grain for ORB keypoints
    grain = rng.normal(0, 3.5, (h // 2, w // 2, 1)).astype(np.float32)
    grain_full = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    room = np.clip(room.astype(np.float32) + grain_full, 0, 255).astype(np.uint8)

    # 2. Lower wood wainscoting / dado rail at bottom
    rail_y = int(0.78 * h)
    room[rail_y:rail_y + 6, :] = (95, 115, 135)
    room[rail_y + 6:, :] = (72, 92, 118)
    for x_plank in range(0, w, 64):
        cv2.line(room, (x_plank, rail_y + 6), (x_plank, h), (58, 76, 98), 1)

    # 3. Framed colored wall poster (high-contrast 4-corner quad for `Auto-Suggest Quad` & surface compensation)
    x0, y0, x1, y1 = 170, 75, 480, 315
    # Dark frame border (gives crisp Canny edges for suggest_quad)
    cv2.rectangle(room, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), (35, 42, 52), -1)
    # Two-tone surface inside frame (warm cream left, terracotta/red right) to showcase surface compensation!
    mid_x = (x0 + x1) // 2
    room[y0:y1, x0:mid_x] = (182, 202, 215)
    room[y0:y1, mid_x:x1] = (88, 96, 178)

    # Subtle decorative geometric motifs on the wall for rich ORB feature descriptors
    for cx, cy in ((85, 130), (85, 240), (555, 130), (555, 240)):
        cv2.circle(room, (cx, cy), 18, (145, 162, 178), 2, cv2.LINE_AA)
        cv2.circle(room, (cx, cy), 6, (125, 142, 158), -1, cv2.LINE_AA)

    return room


class DemoRoomCamera:
    """Drop-in replacement for `cv2.VideoCapture` that streams the synthetic demo room with live sensor noise."""

    def __init__(self, size: Size = DEMO_CAMERA_SIZE):
        self._base = make_demo_room_base(size)
        self._rng = np.random.default_rng(7)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._opened:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        noise = self._rng.integers(-2, 3, size=self._base.shape, dtype=np.int16)
        frame = np.clip(self._base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return True, frame

    def release(self) -> None:
        self._opened = False


def ensure_sample_memories(memories_dir: Path) -> list[Path]:
    """
    Populate `memories_dir` with 3 built-in procedural memory images if no images are present yet,
    so the UI has ready-to-drag memories on first launch.
    """
    memories_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        p for p in memories_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    if existing:
        return sorted(existing)

    w, h = 480, 320
    created: list[Path] = []

    # 1. Mountain Window View (sky gradient, sun, layered mountain ridges, window mullions)
    win = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        win[y, :] = (
            int(235 * (1 - t) + 120 * t),  # B
            int(175 * (1 - t) + 140 * t),  # G
            int(110 * (1 - t) + 245 * t),  # R
        )
    cv2.circle(win, (350, 95), 38, (180, 235, 255), -1, cv2.LINE_AA)
    ridge1 = np.array([[0, 240], [110, 150], [230, 205], [360, 130], [480, 210], [480, h], [0, h]], np.int32)
    ridge2 = np.array([[0, 275], [160, 195], [300, 250], [420, 185], [480, 245], [480, h], [0, h]], np.int32)
    cv2.fillPoly(win, [ridge1], (115, 85, 65), cv2.LINE_AA)
    cv2.fillPoly(win, [ridge2], (75, 55, 42), cv2.LINE_AA)
    # Window pane frame
    cv2.rectangle(win, (0, 0), (w - 1, h - 1), (235, 238, 242), 14)
    cv2.line(win, (w // 2, 0), (w // 2, h), (235, 238, 242), 8)
    cv2.line(win, (0, h // 2), (w, h // 2), (235, 238, 242), 8)
    p1 = memories_dir / "mountain_window.png"
    cv2.imwrite(str(p1), win)
    created.append(p1)

    # 2. Vintage Geometric Tapestry / Rug
    rug = np.full((h, w, 3), (48, 62, 135), dtype=np.uint8)
    cv2.rectangle(rug, (18, 18), (w - 18, h - 18), (195, 215, 235), 4)
    cv2.rectangle(rug, (34, 34), (w - 34, h - 34), (165, 130, 55), -1)
    diamond = np.array([[w // 2, 55], [w - 85, h // 2], [w // 2, h - 55], [85, h // 2]], np.int32)
    cv2.fillPoly(rug, [diamond], (65, 175, 230), cv2.LINE_AA)
    inner_d = np.array([[w // 2, 95], [w - 145, h // 2], [w // 2, h - 95], [145, h // 2]], np.int32)
    cv2.fillPoly(rug, [inner_d], (45, 55, 120), cv2.LINE_AA)
    cv2.circle(rug, (w // 2, h // 2), 26, (215, 230, 245), -1, cv2.LINE_AA)
    p2 = memories_dir / "vintage_tapestry.png"
    cv2.imwrite(str(p2), rug)
    created.append(p2)

    # 3. Botanical Cyanotype Print
    bot = np.full((h, w, 3), (145, 95, 35), dtype=np.uint8)
    cv2.rectangle(bot, (20, 20), (w - 20, h - 20), (235, 230, 220), 3)
    cv2.line(bot, (w // 2, h - 35), (w // 2, 55), (235, 230, 220), 4, cv2.LINE_AA)
    for offset_y, side in ((95, -1), (125, 1), (160, -1), (195, 1), (230, -1)):
        cx = w // 2 + side * 48
        angle = -28 if side < 0 else 28
        cv2.ellipse(bot, (cx, offset_y), (42, 16), angle, 0, 360, (235, 230, 220), -1, cv2.LINE_AA)
    p3 = memories_dir / "cyanotype_print.png"
    cv2.imwrite(str(p3), bot)
    created.append(p3)

    return created
