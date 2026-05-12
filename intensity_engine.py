import cv2
import numpy as np

def clip_intensity(image: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """
    Clips pixel values outside the provided range. 
    Pixels below min_val are set to 0.
    Pixels above max_val are set to 255.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        min_val (int): Minimum intensity threshold.
        max_val (int): Maximum intensity threshold.
        
    Returns:
        np.ndarray: Processed image.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    # Apply transformation: values below min_val -> 0, values above max_val -> 255
    # Values between min_val and max_val remain as they are or are they kept?
    # Usually clipping means keeping the range but here "Pixels below min_val should become 0" 
    # suggests a background suppression for thermal images.
    result = np.copy(image)
    result[image < min_val] = 0
    result[image > max_val] = 255
    
    return result.astype(np.uint8)

def log_transform(image: np.ndarray) -> np.ndarray:
    """
    Applies logarithmic transformation to stretch contrast.
    Formula: s = c * log(1 + r)
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        
    Returns:
        np.ndarray: Log-transformed image in uint8 format.
    """
    # Formula: s = c * log(1 + r)
    # c = 255 / log(1 + max_pixel_value)
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    
    # Specify the data type to be uint8
    log_image = np.array(log_image, dtype=np.uint8)
    
    return log_image

def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Applies standard histogram equalization using OpenCV.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        
    Returns:
        np.ndarray: Histogram equalized image.
    """
    return cv2.equalizeHist(image)
