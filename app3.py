import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# Set up Streamlit page configuration
st.set_page_config(
    page_title='Object Detection using YOLO Model',
    page_icon='🚘',
    layout='centered'
)

# Streamlit app header
st.title('🚘 Object Detection Application')
st.markdown('Upload an image or video and select a YOLO model to detect objects.')

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

# File uploader for images and videos
st.subheader('Upload an Image or Video')
uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG, MP4", type=["jpg", "jpeg", "png", "mp4"])

# Function to process and display image detections
def process_image(image):
    # Ensure the image is RGB
    if len(image.shape) == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the model and plot the results
    results = model(image)
    processed_image = results[0].plot()
    return processed_image

# Function to process and display video detections
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# Handling the uploaded file
if uploaded_file:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Process the image and display the result
        st.subheader('Detection Result')
        processed_image = process_image(image)
        st.image(processed_image, caption='Processed Image', use_column_width=True)

    elif uploaded_file.type == 'video/mp4':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        # Process the video
        st.subheader('Detection Result')
        process_video(video_path)
