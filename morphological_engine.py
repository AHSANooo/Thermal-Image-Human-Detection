import cv2
import numpy as np

def open(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def size_filter(img, stats, pRange):
    valid_people = []
    for i, stat in enumerate(stats):
        x, y, w, h, area, centroid = stat
        if pRange[0] <= area <= pRange[1]:
            valid_people.append({
                'bbox': [x, y, w, h],
                'area': area,
                'center': centroid
            })
        else:
            print(f"Discarding blob {i}: area = {area} is outside the image range {pRange[0]}-{pRange[1]}")
    return valid_people
