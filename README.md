# ReKnew

Desktop memory overlay MVP: pick a surface in the camera feed, drag a memory image onto it, and see the mixed reality result with a Real/Mixed toggle.

## Quick start (desktop UI)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install opencv-python pillow PyQt6
# Optional (for selfie segmentation-assisted plane hinting): pip install mediapipe
python desktop_app.py
```

## How to use

- The app opens a window with the live camera feed and a memory pane (sample images in `assets/images`).
- Click four corners of the surface/plane you want to cover (clockwise or counterclockwise).
- Drag a memory thumbnail onto the camera view (or double-click a thumbnail) to warp it onto that quad.
- Toggle `Show Mixed` / `Show Real` to switch between the raw feed and the overlay.
- `New Capture` resumes the live feed and clears overlays; `Reset Selection` clears the four-corner picks; `Import Memories` loads more images from disk.
- `Auto-Suggest Quad` tries to find a prominent planar contour and fills the selection for you.
- `Color match overlays` adjusts overlay brightness/colors toward the scene underlay for better blending.
- Overlays are persisted with ORB features; when the camera sees the same scene again, overlays reattach automatically to the detected plane.
- Per-overlay sliders: select an overlay from the list on the right, adjust `Opacity` or `Brightness` to reapply it.
- Scenes: use `Save Scene` / `Load Scene` to export/import overlay setups (stored as JSON in `scenes/`).

## Legacy prototype

`main.py` contains the earlier OpenCV-only prototype (click-drag rectangle + Tk file picker).
