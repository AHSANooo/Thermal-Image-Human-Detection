import tempfile
from typing import Tuple

import cv2
import numpy as np
import streamlit as st

from pipeline import pipeline


def _decode_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def _draw_overlay(image: np.ndarray, humans, postures) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, (human, posture) in enumerate(zip(humans, postures), start=1):
        x, y, w, h = map(int, human["bbox"])
        label = f"Human{idx}: {posture}"
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


st.set_page_config(page_title="Thermal Posture Detection", layout="wide")
st.title("Thermal Posture Detection")
st.write("Upload a thermal image to detect people and classify posture.")

with st.sidebar:
    st.header("Pipeline Controls")
    clip_min = st.slider("Clip min", 0, 255, 110)
    clip_max = st.slider("Clip max", 0, 255, 255)
    gaussian_ksize = st.slider("Gaussian kernel size", 1, 15, 5, step=2)
    median_ksize = st.slider("Median kernel size", 1, 15, 5, step=2)
    threshold_method = st.selectbox("Thresholding", ["hysteresis", "otsu"], index=0)
    if threshold_method == "hysteresis":
        hyst_low = st.slider("Hysteresis low", 0.0, 1.0, 0.9, step=0.01)
        hyst_high = st.slider("Hysteresis high", 0.0, 1.0, 0.95, step=0.01)
    else:
        hyst_low = 0.9
        hyst_high = 0.95
    open_ksize_y = st.slider("Open kernel height", 1, 15, 7, step=2)
    open_ksize_x = st.slider("Open kernel width", 1, 15, 3, step=2)
    close_ksize_y = st.slider("Close kernel height", 1, 15, 7, step=2)
    close_ksize_x = st.slider("Close kernel width", 1, 15, 3, step=2)
    size_min = st.slider("Size filter min", 0, 50000, 5000, step=500)
    size_max = st.slider("Size filter max", 0, 100000, 40000, step=1000)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = _decode_image(uploaded_file.read())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    params = {
        "clip_min": clip_min,
        "clip_max": clip_max,
        "gaussian_ksize": gaussian_ksize,
        "median_ksize": median_ksize,
        "threshold_method": threshold_method,
        "hyst_low": hyst_low,
        "hyst_high": hyst_high,
        "open_ksize_y": open_ksize_y,
        "open_ksize_x": open_ksize_x,
        "close_ksize_y": close_ksize_y,
        "close_ksize_x": close_ksize_x,
        "size_min": size_min,
        "size_max": size_max,
    }

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, image)
        result, valid_humans, features, postures = pipeline(temp_file.name, params=params)

    overlay = _draw_overlay(image, valid_humans, postures)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, channels="GRAY", use_container_width=True)
    with col2:
        st.subheader("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    if valid_humans:
        st.subheader("Detected People")
        rows = []
        for idx, (human, feat, posture) in enumerate(zip(valid_humans, features, postures), start=1):
            rows.append({
                "Human": f"Human{idx}",
                "Posture": posture,
                "BBox": human["bbox"],
                "Area": int(human["area"]),
                "Aspect Ratio": float(feat["aspect_ratio"]),
                "Fill Ratio": float(feat["fill_ratio"]),
                "Upper/Lower Ratio": float(feat["upper_lower_ratio"]),
            })
        st.table(rows)
    else:
        st.info("No valid humans detected.")
