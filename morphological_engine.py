import cv2
import numpy as np

def open(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def close(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

