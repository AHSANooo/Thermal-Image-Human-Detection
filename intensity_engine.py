import cv2
import numpy as np

def clip_intensity(image: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """
    Zeros out background pixels while preserving the exact original intensities
    of pixels within [min_val, max_val]. Designed to isolate human heat
    signatures in 8-bit thermal images.

    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        min_val (int): Minimum intensity threshold (inclusive).
        max_val (int): Maximum intensity threshold (inclusive).

    Returns:
        np.ndarray: Masked image with background pixels set to 0, uint8.
    """
    # Vectorised mask: keep original values inside the range, zero everything else
    return np.where(
        (image >= min_val) & (image <= max_val),
        image,
        0
    ).astype(np.uint8)

def log_transform(image: np.ndarray) -> np.ndarray:
    """
    Applies logarithmic transformation to stretch contrast.
    Formula: s = c * log(1 + r), where c = 255 / log(1 + max_pixel).
    Scales the output to the full 8-bit range.

    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.

    Returns:
        np.ndarray: Log-transformed image in uint8 format.
    """
    # Cast to float32 first to avoid uint8 overflow / silent truncation
    img_float = image.astype(np.float32)

    # Guard against all-zero input (e.g., fully masked background)
    if np.max(img_float) == 0:
        return image

    # Scaling constant maps the maximum log value exactly to 255
    c = 255.0 / np.log(1.0 + np.max(img_float))

    # Apply formula and clip to [0, 255] to handle any floating-point overshoot
    log_img = c * np.log(1.0 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)

def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Applies standard histogram equalization using OpenCV.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        
    Returns:
        np.ndarray: Histogram equalized image.
    """
    return cv2.equalizeHist(image)
