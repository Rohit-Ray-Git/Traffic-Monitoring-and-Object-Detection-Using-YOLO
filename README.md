# 🚗 Traffic Monitoring and Object Detection 🚙

This project implements real-time object detection and tracking for traffic monitoring using YOLOv11 (You Only Look Once version 11) and OpenCV. The system tracks various objects (e.g., cars, bicycles, trucks) as they cross a predefined red line on the video feed. The number of objects crossing the line is counted and displayed on the frame.

## Features

- **Real-time object detection and tracking** using YOLOv8.
- **Tracking of multiple object classes**, such as cars, trucks, buses, bicycles, etc.
- **Counts the number of objects** crossing a predefined red line.
- **Displays object IDs, class names**, and bounding boxes on the video feed.
- **Traffic monitoring**: Keeps track of the number of vehicles crossing the red line.
- **Supports custom video input** for testing and monitoring traffic in various scenarios.

## Requirements

Ensure you have the following libraries installed:

- `streamlit` (for UI)
- `opencv-python` (for video processing)
- `ultralytics` (for YOLOv11 model)
- `numpy` (for handling arrays)

Install the dependencies with the following command:

```bash
pip install streamlit opencv-python ultralytics numpy
```

## How to Run
Download the pre-trained YOLOv11 model (yolo11n.pt) or use your own custom-trained model.

Prepare your video or webcam feed. Place it in the working directory.

## Run the Streamlit app:

Ensure you have all the required dependencies installed.
Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

## Adjust Parameters:

- Choose the video input or webcam feed.
- Customize the object classes that YOLO should detect (e.g., vehicles, pedestrians).
- Set the red line position to detect when objects cross.

## Streamlit UI
The Streamlit UI provides a simple and interactive front-end for users to:
- Upload a video.
- Adjust settings like object class selection.
- Visualize the real-time object tracking and counting.


## Code Overview
### 1. Object Detection & Tracking
The object detection is performed using YOLOv8 (You Only Look Once), a deep learning model for fast and accurate object detection. It tracks various objects in each frame, drawing bounding boxes and labels with object IDs. The tracker also monitors when objects cross a predefined red line and counts how many have crossed.

### 2. OpenCV Video Processing
OpenCV is used to handle the video capture, process frames, and display the results in real-time. It overlays bounding boxes, IDs, and counts of detected objects on each frame.

### 3. Streamlit UI
Streamlit is used to create a professional and interactive user interface for selecting video input, visualizing real-time traffic monitoring, and displaying the object detection results.

## Example Screenshots

![Screenshot](https://github.com/user-attachments/assets/18250f00-fc9a-4948-abb7-a5515c40e8d2)

- Red Line: A red line is drawn to monitor when objects cross.
- Object Detection: Detected objects have bounding boxes and class labels.
- Real-Time Count: The count of each class (vehicle, bike, etc.) that crosses the line is updated dynamically.

## Acknowledgements
- YOLOv11: The object detection model used in this project.
- OpenCV: Used for video capture and frame processing.
- Streamlit: For building the interactive web application.
- Ultralytics: For providing the YOLOv8 model implementation.

