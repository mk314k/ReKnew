# main.py

import cv2
from modules.camera import init_camera, get_frame
from modules.display import show_frame, wait_key
from modules.utils import load_image
from modules.image_overlay import overlay_image
from tkinter import Tk, filedialog
import modules.object_selection as obj_sel

def main():
    cap = init_camera()
    window_name = 'ReKnew - Memory Overlay MVP'
    cv2.namedWindow(window_name)

    paused = False  # Freeze on the overlaid frame after a placement
    base_frame = None

    while True:
        if not paused or base_frame is None:
            frame = get_frame(cap)
            base_frame = frame.copy()
        else:
            frame = base_frame.copy()

        # Visualize selection points/polygon on the live frame
        obj_sel.draw_selection(frame)

        # On 4 points, prompt for an image and place it
        if obj_sel.selection_complete:
            root = Tk()
            root.withdraw()  # Hide Tkinter GUI window
            image_path = filedialog.askopenfilename(
                title='Select an Image to Overlay',
                filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]
            )
            root.destroy()

            if image_path:
                overlay_img = load_image(image_path)
                frame = overlay_image(frame, overlay_img, obj_sel.points)
                base_frame = frame.copy()
                paused = True  # Freeze to let user inspect the placement
            else:
                print("No image selected.")

            obj_sel.reset_selection()

        # On-screen instructions
        cv2.putText(
            frame,
            "Click 4 corners of a surface, then choose an image.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Keys: n/new capture, r/reset selection, q/quit",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        status_msg = "Status: frozen on overlay" if paused else "Status: live camera"
        cv2.putText(
            frame,
            status_msg,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if paused else (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        # Show the frame and set mouse callback
        show_frame(window_name, frame)

        key = wait_key()

        if key == ord('r'):
            obj_sel.reset_selection()
            paused = False
            base_frame = None
        elif key == ord('n'):
            obj_sel.reset_selection()
            paused = False
            base_frame = None
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
