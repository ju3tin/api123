import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import uuid
import os

st.set_page_config(page_title="Hand Tracking", layout="wide")
st.title("🖐️ Real-time Hand Tracking with MediaPipe")

st.sidebar.header("Settings")
min_detection = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.8, 0.05)
min_tracking = st.sidebar.slider("Tracking Confidence", 0.5, 1.0, 0.5, 0.05)
save_images = st.sidebar.checkbox("Save Images", False)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=min_detection,
    min_tracking_confidence=min_tracking
)

if save_images and not os.path.exists("output"):
    os.makedirs("output")

run = st.checkbox("🚀 Start Camera", value=False)

if run:
    cap = cv2.VideoCapture(0)
    frame_window = st.empty()
    status = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status.error("Cannot access camera")
            break

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            if save_images:
                cv2.imwrite(f"output/{uuid.uuid1()}.jpg", image)

        # Show frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_window.image(image, channels="RGB", use_column_width=True)

        if st.button("⛔ Stop Camera"):
            break

    cap.release()
else:
    st.info("👆 Check 'Start Camera' to begin")

st.caption("Built with MediaPipe + Streamlit")
