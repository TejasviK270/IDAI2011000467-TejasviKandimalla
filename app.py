import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from parking_logic import summarize

st.set_page_config(page_title="ParkVision AI", layout="wide")

# Update this order to match what data.yaml printed in Step 2
CLASS_NAMES = ["space-empty", "space-occupied"]

@st.cache_resource
def load_model():
    return YOLO("model/parking_yolo.pt")

model = load_model()

st.title("🅿️ ParkVision AI — Intelligent Urban Parking Analytics")
st.write("Upload a parking lot image to detect and count occupied/empty slots.")

uploaded_file = st.file_uploader("Upload parking lot image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(image, conf=0.25, verbose=False)
    result = results[0]

    occupied_count = 0
    empty_count = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        if label == "space-occupied":
            occupied_count += 1
        elif label == "space-empty":
            empty_count += 1

    annotated = result.plot()
    annotated = annotated[:, :, ::-1]  # BGR to RGB

    total_slots = occupied_count + empty_count
    results_summary = summarize(total_slots, occupied_count)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(annotated, caption="Detected parking slots", use_container_width=True)

    with col2:
        st.metric("Total Slots", results_summary["total_slots"])
        st.metric("Occupied Slots", results_summary["occupied_slots"])
        st.metric("Available Slots", results_summary["available_slots"])
        st.metric("Occupancy %", f"{results_summary['occupancy_percent']}%")
        st.write(f"**Congestion level:** {results_summary['congestion_level']}")
        st.info(results_summary["recommendation"])
else:
    st.write("👆 Upload an image to get started.")
