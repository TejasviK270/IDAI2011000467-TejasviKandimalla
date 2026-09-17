import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from parking_logic import summarize

st.set_page_config(page_title="ParkVision AI", layout="wide")

CLASS_ORDER = ["empty", "occupied"]  # matches your trained mapping: {'empty': 0, 'occupied': 1}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/parking_model.keras")

@st.cache_data
def load_positions():
    with open("CarParkPos.pkl", "rb") as f:
        return pickle.load(f)  # now a list of (x, y, w, h)

model = load_model()
posList = load_positions()

st.title("🅿️ ParkVision AI — Intelligent Urban Parking Analytics")
st.write("Upload a parking lot image (same camera angle as the reference image) to see live slot-level occupancy.")

uploaded_file = st.file_uploader("Upload parking lot image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    display_img = img_array.copy()

    occupied_count = 0

    for (x, y, w, h) in posList:
        crop = img_array[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (224, 224)) / 255.0
        crop_resized = np.expand_dims(crop_resized, axis=0)

        pred = model.predict(crop_resized, verbose=0)[0][0]
        label = CLASS_ORDER[1] if pred > 0.5 else CLASS_ORDER[0]

        if label == "occupied":
            occupied_count += 1
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)

    total_slots = len(posList)
    results = summarize(total_slots, occupied_count)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(display_img, caption="Annotated parking lot", use_container_width=True)

    with col2:
        st.metric("Total Slots", results["total_slots"])
        st.metric("Occupied Slots", results["occupied_slots"])
        st.metric("Available Slots", results["available_slots"])
        st.metric("Occupancy %", f"{results['occupancy_percent']}%")
        st.write(f"**Congestion level:** {results['congestion_level']}")
        st.info(results["recommendation"])
else:
    st.write("👆 Upload an image to get started.")
