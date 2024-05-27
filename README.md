# Speed Estimation Using Ultralytics YOLOv8

## Overview

This repository contains code for estimating the speed of vehicles using Ultralytics YOLOv8, a state-of-the-art object detection model. The project utilizes computer vision techniques to track vehicles in a video and estimate their speed based on their movement across predefined regions in the frame.

## Demo

https://github.com/AsadShibli/Speed-Estimation-Using-Ultralytics-YOLOv8/assets/119102237/57d76f84-0186-4c82-a5a1-a3cbe7952379

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- Google Colab (for running the code in a cloud environment)

## Installation

Install the required dependencies:
```bash
!pip install ultralytics
```
## Setup

Mount your Google Drive to access the project files and navigate to the project directory:
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/projects/speed_estimation
```
## Code Snippet

The following code snippet demonstrates how to use YOLOv8 for vehicle speed estimation:
```python
import cv2
from ultralytics import YOLO, solutions

# Load YOLOv8 model
model = YOLO("yolov8l.pt")
names = model.model.names

# Open video file
cap = cv2.VideoCapture("input_video/53125-472583428_small.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer setup
video_writer = cv2.VideoWriter("output_video/speed_estimation6.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define line points for speed estimation
line_pts = [(0, h//2), (w, h//2)]

# Initialize speed estimation object
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

# Process video frames
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```
## Challenges

Implementing speed estimation using computer vision comes with several challenges:

1.**Weather Conditions**: Rain, fog, or snow can obstruct road visibility, making it difficult to track vehicles accurately.

2.**Occlusions**: Other vehicles or objects can block the view, complicating the tracking process.

3.**Lighting Conditions**: Poor lighting, shadows, or glare from the sun can affect the accuracy of speed estimation.

## Accuracy

Speed estimation is an approximation and may not be completely accurate. The estimation can vary depending on factors such as GPU speed and the quality of the input video.

## Acknowledgements

The code snippet was adapted from the Ultralytics documentation. Special thanks to the Ultralytics team for their comprehensive and well-documented object detection models.




