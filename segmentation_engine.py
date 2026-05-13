import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold

def otsu_thresholding(img):
    optimal_threshold_value, resultant_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return resultant_img

def adaptive_thresholding(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def cca(img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    return stats, labels, num_labels, centroids

def hysteresis_thresholding(img, low: float = 0.9, high: float = 0.95):
    return apply_hysteresis_threshold(img, low, high).astype(np.uint8)
