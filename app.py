"""
Streamlit front-end for the custom YOLOv8 detector trained in YOLO_Implementation.ipynb.

Run with:
    streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 Object Detector", page_icon="🥤", layout="wide")

st.title("🥤 YOLOv8 Object Detector")
st.write(
    "Upload an image and the model will draw bounding boxes around every object it detects "
    "(by default, this model was trained to spot **Pepsi Bottles**)."
)

DEFAULT_WEIGHTS = "runs/detect/runs/bottle_detect/exp3/weights/best.pt"


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per weights file)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path: str):
    return YOLO(weights_path)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

weights_source = st.sidebar.radio(
    "Model weights", ["Use a path on disk", "Upload a weights file"]
)

weights_path = None

if weights_source == "Upload a weights file":
    weights_file = st.sidebar.file_uploader("Upload .pt or .onnx weights", type=["pt", "onnx"])
    if weights_file is not None:
        save_dir = Path("uploaded_weights")
        save_dir.mkdir(exist_ok=True)
        weights_path = save_dir / weights_file.name
        weights_path.write_bytes(weights_file.getbuffer())
        weights_path = str(weights_path)
else:
    weights_path = st.sidebar.text_input("Path to weights file (.pt or .onnx)", value=DEFAULT_WEIGHTS)

conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.0005, 0.00005)
iou_threshold = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.0045, 0.00005)

st.sidebar.divider()
st.sidebar.caption(
    "Tip: after training, copy `best.pt` from "
    "`runs/bottle_detect/exp1/weights/` into this app's folder, "
    "or just point the path above at it."
)

# ----------------------------------------------------------------------------
# Validate weights before going further
# ----------------------------------------------------------------------------
if not weights_path:
    st.info("👈 Upload a weights file in the sidebar to get started.")
    st.stop()

if not Path(weights_path).exists():
    st.warning(f"⚠️ No weights file found at `{weights_path}`. Update the path in the sidebar, or upload one.")
    st.stop()

try:
    model = load_model(weights_path)
except Exception as e:
    st.error(f"Failed to load model from `{weights_path}`: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Image upload + inference
# ----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Upload an image to run detection.")
    st.stop()

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner(f"Running detection on {uploaded_file.name}..."):
        results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
    result = results[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    st.subheader(uploaded_file.name)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", width="stretch")
    with col2:
        st.image(annotated, caption="Detections", width="stretch")

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        st.markdown(f"**Found {len(boxes)} object(s):**")
        rows = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
            rows.append(
                {
                    "Class": cls_name,
                    "Confidence": f"{conf:.1%}",
                    "BBox (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})",
                }
            )
        st.table(rows)
    else:
        st.info("No objects detected above the confidence threshold.")

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name=f"detected_{Path(uploaded_file.name).stem}.png",
        mime="image/png",
        key=f"download_{uploaded_file.name}",
    )
    st.divider()