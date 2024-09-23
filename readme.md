# Video Processing and Object Detection Project
![image](https://github.com/user-attachments/assets/50f061df-36cc-4397-b9e7-cd06132054e2)

![image](https://github.com/user-attachments/assets/cf9f4d6b-3e6d-4bbf-9c22-dda803d9f605)

This project consists of two Python scripts aimed at video processing tasks. The first script downloads and trims YouTube videos, while the second script detects and tracks objects within the video using state-of-the-art models for object detection and tracking.

## 1. YouTube Video Downloader and Trimmer (`tube.py`)

### Description:
This Python script allows you to download a YouTube video in 1080p resolution and trim a specific portion for further analysis. The trimmed video can then be used as input for object detection and tracking.

### Requirements:
- Python 3.x
- Libraries: `pytube`, `moviepy`

### Installation:
To install the necessary dependencies, run the following command:
```bash
pip install pytube moviepy
```

### Usage:
1. Run the `tube.py` script to download and trim a video:
```bash
python tube.py
```

This script downloads the video and extracts the portion specified in the script, saving it locally for further processing.

---

## 2. Object Detection and Tracking (`actors_detect.py`)

### Description:
This Python script is designed to detect and track objects in video files using the YOLOv3 model for object detection and MediaPipe for pose and hand tracking. It focuses on detecting humans, analyzing their movements, and identifying specific clothing colors, such as detecting if a person is wearing yellow.

### Requirements:
- Python 3.x
- Libraries: `OpenCV (cv2)`, `NumPy`, `MediaPipe`

### Installation:
1. Install the required libraries using pip:
```bash
pip install opencv-python numpy mediapipe
```

2. Download the necessary YOLOv3 files:
   - [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)
   - [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

   Place these files in the same directory as `actors_detect.py`.

### Usage:
1. Set the `video_file` variable in the script to the path of your input video file.
2. Run the script:
```bash
python actors_detect.py
```
3. The script will process the video, display it with real-time object detection and tracking, and save the output as `output_video.mp4`. Press 'q' to quit the display.

### Functionality:
- **Object Detection**: The script uses the YOLOv3 model to detect objects (specifically people) in each video frame.
- **Tracking**: For each detected person, MediaPipe is used to track pose and hand landmarks.
- **Color Detection**: It detects whether the person is wearing a yellow dress. If detected, the bounding box is colored pink; otherwise, it's green.
- **Output**: The video is displayed in real-time and saved as `output_video.mp4` with bounding boxes and tracking markers.

### Notes:
- Ensure the `yolov3.weights`, `yolov3.cfg`, and `coco.names` files are in the same directory as the script.
- GPU Acceleration: The script is configured to use CUDA for GPU acceleration. If you don't have a compatible GPU, remove the following lines:
  ```python
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  ```
- **Confidence Threshold**: The script uses a confidence threshold of 0.5 for object detection. You can modify this by changing the line:
  ```python
  if confidence > 0.5 and class_id == 0:
  ```

### Example Output:
The output video will show detected individuals with bounding boxes, colored based on clothing detection, and include pose and hand tracking landmarks.

---

### Project Workflow:
1. **Download and Trim Video**: Use `tube.py` to download and trim the necessary YouTube video segment.
2. **Object Detection and Tracking**: Process the trimmed video using `actors_detect.py` for object detection, tracking, and attire analysis.
3. **Output Video**: The final output will be saved as `output_video.mp4`, which contains the analyzed and annotated video with tracking information.

This project is suitable for use cases like video analysis, automated tracking, and behavioral study in videos where object detection and tracking are required.
