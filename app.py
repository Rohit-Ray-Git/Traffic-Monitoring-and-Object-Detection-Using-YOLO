import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Get the class labels from the model
class_list = model.names

# Streamlit UI setup
st.set_page_config(page_title="🚗 Traffic Monitoring and Object Detection 🚙", layout="centered")

st.title("🚗 Traffic Monitoring and Object Detection 🚙")
st.markdown("""
    This application uses **YOLOv8** to detect and classify vehicles in real-time. 
    Track the movement of vehicles, count their types, and analyze traffic flow. 
    Upload a video for monitoring.
""", unsafe_allow_html=True)

# Main screen content
st.subheader("🚦 Choose a Video File 🚦")

# Create button for video upload
uploaded_video = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])

# Display the processing message
if uploaded_video is not None:
    st.text("Processing video...")

    # Initialize video capture using the uploaded file
    cap = cv2.VideoCapture(uploaded_video.name)

    line_y_red = 430  # red line position
    class_counts = defaultdict(int)
    crossed_ids = set()

    # Create a placeholder for displaying the video
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 on the frame
        results = model.track(frame, persist=True, classes=[0, 1, 2, 3, 5, 6, 7])

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.numpy()  # Converting box coordinates to numpy
            track_ids = results[0].boxes.id.int().tolist()  # Convert to list for track IDs
            class_indices = results[0].boxes.cls.int().tolist()  # Convert to list for class IDs
            confidences = results[0].boxes.conf.cpu().numpy()  # Convert confidence to numpy array

            # Draw red line on the frame
            cv2.line(frame, (0, line_y_red), (frame.shape[1], line_y_red), (0, 0, 255), 3)

            # Loop through each detected object
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                class_name = class_list[class_idx]

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Id: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Check if the object has crossed the red line
                if cy > line_y_red and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

            # Display the counts on the frame
            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30

            # Convert frame for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width = True)

    cap.release()
    cv2.destroyAllWindows()

else:
    st.warning("Please upload a video file to begin processing. 🎥")
