import cv2
import numpy as np
import matplotlib.pyplot as plt

import morphological_engine
from intensity_engine import clip_intensity, log_transform, equalize_histogram, apply_mask
from spatial_engine import gaussian_smooth
from segmentation_engine import otsu_thresholding, cca, hysteresis_thresholding

from spatial_engine import gaussian_smooth, median_filter, laplacian_sharpening
from morphological_engine import open, close, size_filter
from feature_extraction import extract_features, classify_posture

def pipeline(img_path, params=None):
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        raise FileNotFoundError(f"Could not load image at '{img_path}'. Ensure the file exists.")

    params = params or {}
    clip_min = int(params.get("clip_min", 110))
    clip_max = int(params.get("clip_max", 255))
    gaussian_ksize = int(params.get("gaussian_ksize", 5))
    median_ksize = int(params.get("median_ksize", 5))
    hyst_low = float(params.get("hyst_low", 0.9))
    hyst_high = float(params.get("hyst_high", 0.95))
    threshold_method = params.get("threshold_method", "hysteresis")
    open_ksize = (int(params.get("open_ksize_y", 7)), int(params.get("open_ksize_x", 3)))
    close_ksize = (int(params.get("close_ksize_y", 7)), int(params.get("close_ksize_x", 3)))
    size_min = int(params.get("size_min", 5000))
    size_max = int(params.get("size_max", 40000))

    print(f"Loaded  : {img_path}")
    print(f"Shape   : {original_img.shape} | dtype: {original_img.dtype}")
    print(f"Range   : min={original_img.min()}, max={original_img.max()}")

    clipped_img = clip_intensity(original_img, clip_min, clip_max)

    log_img = log_transform(clipped_img)

    equalized_img = equalize_histogram(log_img)

    assert equalized_img.dtype == np.uint8, "Output dtype is not uint8!"
    assert 0 <= equalized_img.min() and equalized_img.max() <= 255, "Pixel values out of [0,255]!"
    print("\n✓ Pipeline complete — all outputs correctly bounded within [0, 255].")

    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
    if median_ksize % 2 == 0:
        median_ksize += 1

    smoothed_img = gaussian_smooth(equalized_img, kernel_size=(gaussian_ksize, gaussian_ksize))

    median_img = median_filter(smoothed_img, kernel_size=median_ksize)

    sharpened_img = laplacian_sharpening(median_img)

    if threshold_method == "otsu":
        thresholded_img = otsu_thresholding(sharpened_img)
    else:
        thresholded_img = hysteresis_thresholding(sharpened_img, low=hyst_low, high=hyst_high)

    kernel_open = np.ones(open_ksize, np.uint8)
    opened_img = open(thresholded_img, kernel_open)

    kernel_close = np.ones(close_ksize, np.uint8)
    closed_img = close(opened_img, kernel_close)

    stats, labels, num_labels, centroids = cca(closed_img)

    final_img, valid_humans = size_filter(closed_img, stats, (size_min, size_max), labels, num_labels, centroids)

    result = apply_mask(original_img, final_img)

    features = [extract_features(human) for human in valid_humans]
    postures = [classify_posture(feat) for feat in features]

    return result, valid_humans, features, postures
