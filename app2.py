import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from collections import defaultdict

# Set up Streamlit page configuration
st.set_page_config(
    page_title='Vehicle Detection & Counting',
    page_icon='🚘',
    layout='centered'
)

# Streamlit app header
st.title('🚘 Vehicle Detection & Counting')
st.markdown('Upload a video, and the app will count vehicles crossing a line.')

# Center-align content
st.markdown(
    """
    <style>
    .block-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model selection dropdown
st.subheader('Select YOLO Model')
model_options = [
    'yolo11s.pt', 'yolo11n.pt', 'yolov10n.pt', 'yolov10s.pt', 'yolov9s.pt', 'yolov8n.pt', 'yolov8s.pt',
    'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
]
selected_model = st.selectbox('Choose a YOLO model:', model_options)

# Load the selected YOLO model
model = YOLO(selected_model)
st.success(f'Selected Model: {selected_model}')

# Define vehicle classes to count
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# File uploader for videos
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# Function to count vehicles crossing a line
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    # Define counting line position
    line_position = 430  # Y-coordinate of the line
    vehicle_count = defaultdict(int)  # Dictionary to store counts for each vehicle type

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using YOLO
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes and classes
        frame_height, frame_width = frame.shape[:2]

        # Draw counting line
        cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 255, 0), 2)

        # Process each detection
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            object_type = model.names[int(cls)]

            # Filter for vehicle types only
            if object_type in vehicle_classes:
                # Check if the object crosses the line
                object_center_y = int((y1 + y2) / 2)
                if line_position - 5 <= object_center_y <= line_position + 5:
                    vehicle_count[object_type] += 1

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{object_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the counts in the top-left corner
        y_offset = 20
        for vehicle, count in vehicle_count.items():
            cv2.putText(frame, f"{vehicle}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# Handle uploaded file
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Process the video
    st.subheader('Detection Result')
    process_video(video_path)
