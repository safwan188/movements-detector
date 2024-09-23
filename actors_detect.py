import cv2
import numpy as np
import mediapipe as mp

# Load the pre-trained object detection model (YOLOv3)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input video file
video_file = "trimmed_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a video writer object
output_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

# Initialize MediaPipe pose and hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the model
    net.setInput(blob)

    # Get the output layers of the model
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass through the model
    layer_outputs = net.forward(output_layers)

    # Initialize lists to store the detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process the output layers
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Check if the detected object is a person
            if label == "person":
                # Extract the region of interest (ROI) containing the person
                roi = frame[y:y+h, x:x+w]

                # Convert the ROI to RGB
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # Detect pose landmarks
                pose_results = pose.process(roi_rgb)

                # Detect hand landmarks
                hand_results = hands.process(roi_rgb)

                # Draw pose landmarks on the ROI
                if pose_results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Draw hand landmarks on the ROI
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert the ROI to HSV color space
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define the range of yellow color in HSV
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([40, 255, 255])

                # Create a mask for yellow color
                mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

                # Check if the yellow color is present in the ROI
                if cv2.countNonZero(mask) > 0:
                    # If yellow color is detected, use a different color for the bounding box
                    color = (255, 192, 203)  # Red color for the person with yellow dress
                else:
                    color = (0, 255, 0)  # Green color for other persons
            else:
                color = (0, 255, 0)  # Green color for non-person objects

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()