tube.py:
# YouTube Video Downloader and Trimmer

This code provides a Python script to download a YouTube video in 1080p resolution and trim a specific portion of the video.
## Requirements

- Python 3.x
- pytube
- moviepy
## Installation

1. Install the required libraries using pip:

pip install pytube moviepy
## Usage

1. run python tube.py


actors_detect.py:

# Object Detection and Tracking

This code provides a Python script to perform object detection and tracking in a video file. It uses the YOLOv3 model for object detection and MediaPipe for pose and hand tracking.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- MediaPipe

## Installation
1. Install the required libraries using pip:

pip install opencv-python numpy mediapipe



2. The YOLOv3 weights, configuration, and class names files were downloaded from:
- `yolov3.weights`: https://pjreddie.com/media/files/yolov3.weights
- `yolov3.cfg`: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
- `coco.names`: https://github.com/pjreddie/darknet/blob/master/data/coco.names

Place these files in the same directory as the script file.

## Usage

1. Set the `video_file` variable in the script to the path of your input video file.

2. Run the script:
python actors_detect.py
3. The script will process the video and display the output with object detection and tracking. Press 'q' to quit the script.

4. The processed video will be saved as `output_video.mp4` in the same directory.

## Functionality

- The script uses the YOLOv3 model to detect objects in each frame of the video.
- It specifically focuses on detecting persons and draws bounding boxes around them.
- For each detected person, the script uses MediaPipe to detect pose and hand landmarks within the bounding box.
- The script also performs color detection to identify if a person is wearing a yellow dress. If a yellow dress is detected, the bounding box color is set to pink; otherwise, it is set to green.
- The processed video is displayed in real-time and saved as `output_video.mp4`.

## Notes

- Make sure to have the required files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`) in the same directory as the script.
- The script is set to use CUDA for GPU acceleration. If you don't have a compatible GPU, you can remove the lines `net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)` and `net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)`.
- The script uses a confidence threshold of 0.5 for object detection. You can adjust this threshold by modifying the condition `if confidence > 0.5 and class_id == 0:`.
