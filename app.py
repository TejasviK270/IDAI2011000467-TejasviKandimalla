import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from parking_logic import summarize

st.set_page_config(page_title="ParkVision AI", layout="wide")

CLASS_NAMES = ["space-empty", "space-occupied"]

@st.cache_resource
def load_model():
    return YOLO("model/parking_yolo.pt")

model = load_model()

st.title("🅿️ ParkVision AI")
st.subheader("Intelligent Urban Parking Analytics & Space Optimisation")
st.write("Upload a photo of a parking lot to instantly see which spaces are occupied and which are free.")

with st.sidebar:
    st.header("⚙️ Advanced Settings")
    conf_threshold = st.slider(
        "Detection sensitivity",
        min_value=0.05, max_value=0.9, value=0.15, step=0.05,
        help="Lower values detect more slots but may include false positives."
    )
    st.caption("Default settings work well for most images — only adjust if slots are being missed.")

uploaded_file = st.file_uploader("📤 Upload a parking lot image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("Analyzing parking lot..."):
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=0.3,
            imgsz=1920,
            max_det=2000,
            verbose=False,
        )
    result = results[0]

    occupied_count = 0
    empty_count = 0
    display_img = img_array.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "space-occupied":
            occupied_count += 1
            color = (255, 0, 0)
        elif label == "space-empty":
            empty_count += 1
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)

        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 3)

    total_slots = occupied_count + empty_count
    results_summary = summarize(total_slots, occupied_count)

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            display_img,
            caption="🟢 Empty slot   🔴 Occupied slot",
            width="stretch",
        )

    with col2:
        st.markdown("### 📊 Parking Summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total", results_summary["total_slots"])
        m2.metric("Occupied", results_summary["occupied_slots"])
        m3.metric("Available", results_summary["available_slots"])

        st.metric("Occupancy", f"{results_summary['occupancy_percent']}%")

        congestion = results_summary["congestion_level"]
        if congestion == "Low":
            st.success(f"🟢 Congestion Level: **{congestion}**")
        elif congestion == "Moderate":
            st.warning(f"🟡 Congestion Level: **{congestion}**")
        else:
            st.error(f"🔴 Congestion Level: **{congestion}**")

        st.info(f"💡 {results_summary['recommendation']}")

else:
    st.info("👆 Upload an image above to get started.")
    st.caption("Works best with aerial or elevated views of parking lots, similar to the PKLot dataset used for training.")
