# modules/display.py

import cv2
from modules.object_selection import click_and_select


def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, click_and_select)


def wait_key(delay=1):
    return cv2.waitKey(delay) & 0xFF
