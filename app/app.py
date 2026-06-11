import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import random
import time
import folium
from streamlit_folium import st_folium

# ===============================
# FIX LSTM ISSUE
# ===============================
from tensorflow.keras.layers import LSTM

class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# ===============================
# LOAD MODEL + ENCODER
# ===============================
MODEL_PATH = os.path.join("models", "insect_cnn_lstm.h5")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"LSTM": FixedLSTM}
)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# ===============================
# UI
# ===============================
st.title("🌾 Smart Pest Detection System")

FIELD_LENGTH_FT = 200
FIELD_WIDTH_FT = 150
FIELD_AREA = FIELD_LENGTH_FT * FIELD_WIDTH_FT

st.sidebar.header("🌾 Field Info")
st.sidebar.write(f"Length: {FIELD_LENGTH_FT} ft")
st.sidebar.write(f"Width: {FIELD_WIDTH_FT} ft")
st.sidebar.write(f"Area: {FIELD_AREA} sq ft")

base_lat = 12.9726
base_lon = 77.5956

ft_to_lat = 0.0000025
ft_to_lon = 0.0000025

zones = ["Zone A", "Zone B", "Zone C", "Zone D"]

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, 40, 1)

def simulate_sensor():
    return random.uniform(0.2, 0.95)

# ===============================
# MENU
# ===============================
mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Audio", "Live Sensor Simulation", "Farmer Risk Map"]
)

# ===============================
# MODE 1: AUDIO
# ===============================
if mode == "Upload Audio":

    st.subheader("🐛 Pest Detection from Audio")

    file = st.file_uploader("Upload WAV file", type=["wav"])

    if file:

        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        st.audio(temp_path)

        features = extract_features(temp_path)

        prediction = model.predict(features)
        pred_class = np.argmax(prediction)

        if pred_class < len(le.classes_):
            label = le.inverse_transform([pred_class])[0]
        else:
            label = "Unknown Pest"

        confidence = np.max(prediction)
        zone = random.choice(zones)

        st.success(f"🐛 Pest Detected: {label}")
        st.info(f"📍 Zone: {zone}")

        if confidence > 0.8:
            st.error("🔴 HIGH RISK")
        elif confidence > 0.5:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

# ===============================
# MODE 2: SENSOR
# ===============================
elif mode == "Live Sensor Simulation":

    st.subheader("📡 Sensor Simulation Dashboard")

    if st.button("Start Simulation"):

        placeholder = st.empty()

        for _ in range(10):

            with placeholder.container():

                for z in zones:

                    risk = simulate_sensor()

                    if risk > 0.75:
                        status = "🔴 HIGH"
                    elif risk > 0.5:
                        status = "🟡 MEDIUM"
                    else:
                        status = "🟢 LOW"

                    st.write(f"{z} → Risk: {risk:.2f} → {status}")

                time.sleep(1)

# ===============================
# MODE 3: MAP
# ===============================
else:

    st.subheader("🌾 Farmer-Friendly Pest Risk Map")

    risk_map = [
        [0.9, 0.7, 0.3, 0.2],
        [0.8, 0.6, 0.4, 0.3],
        [0.7, 0.5, 0.2, 0.1],
        [0.6, 0.4, 0.3, 0.2]
    ]

    rows = 4
    cols = 4

    m = folium.Map(location=[base_lat, base_lon], zoom_start=18)

    boundary = [
        [base_lat, base_lon],
        [base_lat + FIELD_LENGTH_FT * ft_to_lat, base_lon],
        [base_lat + FIELD_LENGTH_FT * ft_to_lat, base_lon + FIELD_WIDTH_FT * ft_to_lon],
        [base_lat, base_lon + FIELD_WIDTH_FT * ft_to_lon],
        [base_lat, base_lon]
    ]

    folium.Polygon(
        locations=boundary,
        color="blue",
        weight=3,
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    def get_color(risk):
        if risk > 0.75:
            return "red"
        elif risk > 0.5:
            return "orange"
        else:
            return "green"

    for i in range(rows):
        for j in range(cols):

            lat = base_lat + (i / (rows - 1)) * FIELD_LENGTH_FT * ft_to_lat
            lon = base_lon + (j / (cols - 1)) * FIELD_WIDTH_FT * ft_to_lon

            risk = risk_map[i][j]
            color = get_color(risk)

            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=f"Risk Level: {risk:.2f}"
            ).add_to(m)

    st_folium(m, width=700, height=500)