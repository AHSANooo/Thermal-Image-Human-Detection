import cv2
import matplotlib.pyplot as plt
from intensity_engine import clip_intensity, log_transform, equalize_histogram
from spatial_engine import gaussian_smooth, median_filter, laplacian_sharpening

def test_pipeline():
    image_path = 'dataset/04353.jpg'
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        print(f"Failed to load image at {image_path}")
        return
        
    print("Running Stage 1...")
    # --- Stage 1: Intensity Transformation ---
    CLIP_MIN = 110
    CLIP_MAX = 255
    clipped_img = clip_intensity(original_img, min_val=CLIP_MIN, max_val=CLIP_MAX)
    log_img = log_transform(clipped_img)
    eq_img = equalize_histogram(log_img)
    
    print("Running Stage 2...")
    # --- Stage 2: Spatial Filtering ---
    # 1. Gaussian smooth (suppress sensor noise, preserve blob edges)
    smoothed_img = gaussian_smooth(eq_img, kernel_size=(5, 5))
    
    # 2. Median filter (remove salt-and-pepper noise)
    median_img = median_filter(smoothed_img, kernel_size=5)
    
    # 3. Laplacian sharpening (enhance boundaries)
    sharpened_img = laplacian_sharpening(median_img)
    
    # Save the final exact output
    cv2.imwrite("stage2_final_output.png", sharpened_img)
    
    # Plotting a comparison to visualize the pipeline
    print("Saving visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    titles = [
        "0. Original Image", 
        "1. Stage 1 Final (Equalized)", 
        "2. Stage 2 (Gaussian Smoothed)", 
        "3. Stage 2 (Median Filtered)", 
        "4. Stage 2 Final (Laplacian Sharpened)"
    ]
    
    images = [
        original_img, eq_img, smoothed_img,
        median_img, sharpened_img
    ]
    
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    axes[-1].axis('off') # Hide the 6th empty subplot
    
    plt.tight_layout()
    plt.savefig('pipeline_output_stage2.png')
    print("Testing complete. Visuals saved to 'pipeline_output_stage2.png'.")

if __name__ == "__main__":
    test_pipeline()
