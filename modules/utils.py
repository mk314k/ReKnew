# modules/utils.py

from PIL import Image
import cv2
import numpy as np

def load_image(image_path):
    pil_image = Image.open(image_path).convert("RGBA")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
