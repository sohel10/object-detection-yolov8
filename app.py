import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load model once (with fallback to CPU)
@st.cache_resource
def load_model(weights="yolov8n.pt"):
    return YOLO(weights)

model = load_model()

st.title("🚗 License Plate Recognition (YOLOv8 + EasyOCR)")

# Upload image or video
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Handle images
    if uploaded_file.type.startswith("image/"):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save temp file for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model(tmp.name, device="cpu")  # force CPU since GPU not supported yet
            res = results[0]

            # Show annotated image
            annotated = res.plot()
            st.image(annotated, caption="Detection Result", use_column_width=True)

    # Handle videos
    elif uploaded_file.type == "video/mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            results = model(tmp.name, device="cpu", save=True)

            st.success("✅ Detection complete. Saved results to `runs/detect/predict/`")
            st.video(tmp.name)
