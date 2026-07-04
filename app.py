import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="Hand Tracking", layout="wide")
st.title("🖐️ MediaPipe Hands Tracking")
st.write("Real-time hand detection with MediaPipe")

# Sidebar
st.sidebar.header("Settings")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.5, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.5, 1.0, 0.5)
save_images = st.sidebar.checkbox("Save Images", value=False)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)

# Create folder for saved images
if save_images and not os.path.exists("output_images"):
    os.makedirs("output_images")

# Webcam
run = st.checkbox("Start Camera", value=False)

if run:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # Detection
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            # Save image if enabled
            if save_images:
                cv2.imwrite(f"output_images/{uuid.uuid1()}.jpg", image)

        # Display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(image, channels="RGB", use_column_width=True)

        if st.button("Stop"):
            break

    cap.release()

else:
    st.info("Click 'Start Camera' to begin hand tracking")

# Footer
st.caption("Built with MediaPipe + Streamlit")
