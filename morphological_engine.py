import cv2
import numpy as np

def open(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def size_filter(img, stats, pRange, labels, num_labels, centroids=None):
    # Initialize an empty black image
    human_img = np.zeros_like(img)
    valid_people = []

    # Start loop at 1 to skip the background (index 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Check if the component falls within the allowed size range
        if pRange[0] <= area <= pRange[1]:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            component_mask = (labels == i).astype(np.uint8) * 255
            bbox_mask = component_mask[y:y + h, x:x + w]

            valid_people.append({
                'bbox': [x, y, w, h],
                'area': area,
                'mask': bbox_mask,
                'label_idx': i,
                'center': tuple(centroids[i]) if centroids is not None else None
            })

            # Use the index 'i' to pull ONLY this component from the labels map
            human_img[labels == i] = 255
        else:
            print(f"Discarding blob {i}: area = {area} is outside range {pRange[0]}-{pRange[1]}")

    return human_img, valid_people
