import streamlit as st
from PIL import Image
import pickle
import os

st.set_page_config(page_title="Mark Parking Slots")
st.title("🅿️ Mark Parking Slots on Reference Image")

WIDTH = st.sidebar.number_input("Slot width (px)", value=60)
HEIGHT = st.sidebar.number_input("Slot height (px)", value=40)
POS_FILE = "CarParkPos.pkl"

if "posList" not in st.session_state:
    if os.path.exists(POS_FILE):
        with open(POS_FILE, "rb") as f:
            st.session_state.posList = pickle.load(f)
    else:
        st.session_state.posList = []

img = Image.open("reference_lot.jpg")
st.write(f"Image size: {img.size[0]} x {img.size[1]}")

col1, col2 = st.columns(2)
with col1:
    x = st.number_input("X coordinate (top-left)", min_value=0, max_value=img.size[0], value=0)
with col2:
    y = st.number_input("Y coordinate (top-left)", min_value=0, max_value=img.size[1], value=0)

if st.button("Add slot at this position"):
    st.session_state.posList.append((int(x), int(y)))
    with open(POS_FILE, "wb") as f:
        pickle.dump(st.session_state.posList, f)

if st.button("Remove last slot"):
    if st.session_state.posList:
        st.session_state.posList.pop()
        with open(POS_FILE, "wb") as f:
            pickle.dump(st.session_state.posList, f)

from PIL import ImageDraw
preview = img.copy()
draw = ImageDraw.Draw(preview)
for (px, py) in st.session_state.posList:
    draw.rectangle([px, py, px + WIDTH, py + HEIGHT], outline="magenta", width=2)

st.image(preview, caption=f"{len(st.session_state.posList)} slots marked", use_column_width=True)
st.write("Marked positions:", st.session_state.posList)
