import numpy as np

def extract_features(human):
    """Extract geometric features from a single human silhouette"""
    x, y, w, h = human['bbox']
    mask = human['mask']
    area = human['area']

    # Feature 1: Aspect ratio
    aspect_ratio = h / w if w > 0 else 0

    # Feature 2: Fill ratio
    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0

    # Feature 3: Upper/lower mass ratio (for bending detection)
    mid_y = y + h // 2
    upper_mask = mask[0:h//2, :]  # Top half of bounding box
    lower_mask = mask[h//2:, :]   # Bottom half
    upper_mass = np.count_nonzero(upper_mask)
    lower_mass = np.count_nonzero(lower_mask)
    upper_lower_ratio = upper_mass / lower_mass if lower_mass > 0 else 0

    return {
        'aspect_ratio': aspect_ratio,
        'fill_ratio': fill_ratio,
        'upper_lower_ratio': upper_lower_ratio
    }

def classify_posture(features):
    """Rules-based posture classification using thresholds you find"""
    ar = features['aspect_ratio']
    fr = features['fill_ratio']
    ulr = features['upper_lower_ratio']

    # LYING: wide and fills box
    if ar < 1.0 and fr > 0.7:
        return "lying"

    # BENDING: more mass in upper half with compact aspect ratio
    if ulr > 1.6 and fr > 0.4 and ar < 2.0:
        return "bending"

    # TALL SITTING: tall but high fill and upper mass
    if ar > 2.0 and fr >= 0.5 and ulr > 1.1:
        return "sitting"

    # STANDING: clearly tall, or tall with low fill
    if ar > 2.3 or (ar > 2.0 and fr < 0.5):
        return "standing"

    # SITTING: everything else in the middle
    return "sitting"
