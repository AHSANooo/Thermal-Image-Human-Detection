import cv2
import numpy as np

def gaussian_smooth(image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma_x: float = 0.0) -> np.ndarray:
    """
    Applies Gaussian smoothing to suppress sensor noise while attempting to preserve blob edges.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        kernel_size (tuple[int, int]): Size of the Gaussian kernel. Both dimensions must be positive and odd.
        sigma_x (float): Gaussian kernel standard deviation in X direction. If 0, it's calculated from kernel size.
        
    Returns:
        np.ndarray: Smoothed image in uint8 format.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)

def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies a non-linear median filter to remove salt-and-pepper noise.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        kernel_size (int): Size of the aperture. Must be odd and greater than 1 (e.g., 3, 5, 7).
        
    Returns:
        np.ndarray: Median filtered image in uint8 format.
    """
    return cv2.medianBlur(image, kernel_size)

def laplacian_sharpening(image: np.ndarray) -> np.ndarray:
    """
    Enhances human blob boundaries post-smoothing using a Laplacian-based sharpening filter.
    
    Args:
        image (np.ndarray): 8-bit single-channel grayscale image.
        
    Returns:
        np.ndarray: Sharpened image in uint8 format.
    """
    # A standard sharpening kernel based on the Laplacian operator.
    # The center weight is 5 and 4-neighbors are -1. 
    # This applies: Original Image - Laplacian = Sharpened Image
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    
    # filter2D with depth -1 ensures the output has the same depth as the source.
    # OpenCV automatically handles value saturation (clipping to 0-255) for uint8 types.
    return cv2.filter2D(image, -1, kernel)
