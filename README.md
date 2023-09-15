# ReKnew

Bring objects from a past place into the room you are in now: a projector mounted with a camera
lights up your current surfaces with memories (old wallpaper, a rug, a window view, photos).

Without a projector, the app simulates one. The simulation models projection physically: a projector
can only add light, so dark content and colored or dark surfaces limit what can be shown, and room
light washes it out. The same code drives a real projector connected as a second display.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
reknew  # or: python -m reknew (use `reknew --demo` for the built-in synthetic room)
```

Put your memory photos in `memories/` (it stays out of git; 3 sample memories are generated automatically on first run if `memories/` is empty) or add them with `Import Memories`. If no webcam is connected, ReKnew automatically falls back to a built-in synthetic room so you can test surface selection, radiometric compensation, and calibration immediately.

Run the tests with `pip install -e ".[dev]" && pytest`.

On Ubuntu desktop, Qt also needs `sudo apt install libxcb-cursor0`.

## Code layout

| Module | Responsibility |
|---|---|
| `reknew/light.py` | sRGB/linear conversion, surface reflectance estimate |
| `reknew/compositing.py` | Corner ordering, warping memories onto quads, layer compositing |
| `reknew/projection.py` | Projector light model, compensation target and solve |
| `reknew/calibration.py` | Chessboard pattern, projector homography and brightness measurement |
| `reknew/projectors.py` | `Projector` interface and `SimulatedProjector` |
| `reknew/scene.py` | `Overlay`, `RoomFeatures` (recognizing the room again), scene files |
| `reknew/session.py` | `ProjectionSession`: all per-frame state and logic, no UI |
| `reknew/surface_detection.py` | Auto-suggested quads |
| `reknew/demo.py` | Synthetic demo room camera and sample memory generator |
| `reknew/ui/` | Qt window, widgets and the real-projector (`ScreenProjector`) output |

## How to use

- The main view shows what the camera would see with the projector on (`Show Camera` switches to the raw feed).
  The orange outline is the area the projector can reach.
- Click four corners of a surface in any order, then drag a memory thumbnail onto the view (or double-click it).
- `Projector output` (bottom right) is the exact image sent to the projector.
- `Compensate for surface` pre-distorts the image so surface colors and textures show through less.
  `Uniformity` trades evenness across differently colored surfaces against contrast.
  Magenta marks areas the projector physically can't reproduce (darker than the room, colors the surface absorbs, or outside the projector's reach).
- Simulation sliders: `Room light` dims the room, `Projector brightness` sets the virtual projector's strength.
- `Calibrate` projects a chessboard, finds where the projector lands in the camera image and measures its brightness.
  In simulation, it reports the error against the virtual projector's true position.
- Per-overlay `Opacity` / `Brightness`: select an overlay in the list on the right.
- `Save Scene` / `Load Scene` store overlays as JSON in `scenes/`, together with features of the room view.
  On load, the saved view is found in the current camera image and the overlays move with it.

### Real projector

1. Connect the projector as a second display and mount it rigidly next to the camera.
2. Choose `Real projector`. The output opens fullscreen on the last connected display.
3. Aim at a light, flat area and click `Calibrate`. This also captures the room with the projector dark;
   use `Capture Room` to refresh that after lighting changes. Lock the camera's auto-exposure if your webcam allows it.
