import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load YOLO weights and config files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO labels (names of objects YOLO can detect)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up video capture
cap = cv2.VideoCapture("input_video.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define output video properties (for vertical format, 9:16 aspect ratio)
out_width = 720
out_height = 1280
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (out_width, out_height))

# Kalman Filter setup for tracking action smoothing
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)

# Function to process each frame
def process_frame(frame, kalman):
    height, width, _ = frame.shape

    # Create blob from the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detection information
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate weak overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # If any objects are detected, crop around them
    if len(indexes) > 0:
        # Get largest box (assuming the largest detected object is the main focus)
        largest_box = max([boxes[i] for i in indexes.flatten()], key=lambda b: b[2]*b[3])

        # Track and smooth the action using Kalman Filter
        (x, y, w, h) = largest_box
        measured = np.array([[np.float32(x + w // 2)], [np.float32(y + h // 2)]])
        kalman.correct(measured)
        predicted = kalman.predict()

        # Use the Kalman filter's predicted center point to determine the crop area
        pred_x = int(predicted[0] - w // 2)
        pred_y = int(predicted[1] - h // 2)
        cropped_frame = frame[max(0, pred_y):min(height, pred_y + h), max(0, pred_x):min(width, pred_x + w)]

        # Resize the cropped frame to vertical format (9:16)
        resized_frame = cv2.resize(cropped_frame, (out_width, out_height))

        return resized_frame

    # If no object detected, just center crop
    start_x = (width - out_width) // 2
    start_y = (height - out_height) // 2
    return frame[start_y:start_y + out_height, start_x:start_x + out_width]

# Function to process frames in parallel (for multi-threading)
def process_video_parallel(cap, kalman, out):
    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Submit each frame to be processed in a separate thread
            futures.append(executor.submit(process_frame, frame, kalman))
        
        # Write each processed frame to the output video
        for future in futures:
            processed_frame = future.result()
            out.write(processed_frame)

# Process the video using multiple threads
process_video_parallel(cap, kalman, out)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()