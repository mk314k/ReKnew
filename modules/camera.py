# modules/camera.py

import cv2

def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame from camera")
    return frame
