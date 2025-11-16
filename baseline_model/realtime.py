from ultralytics import YOLO
from PIL import Image
import numpy as np
import pytesseract
import cv2

# Load the model
model = YOLO('aadhar_yolo_weights.pt')  # pretrained YOLOv8n model
cap = cv2.VideoCapture(0)  # Open the webcam

# Define classes for AADHAAR number, date of birth, gender, name, address
classes = {0: 'Aadhar Number', 1: 'DOB', 2: 'Gender', 3: 'Name', 4: 'Address'}

capture_frame = False

while True:
    ret, frame = cap.read()  # Read a frame from the video stream
    if not ret:
        break
    
    # Perform inference
    prediction = model.predict(frame, imgsz=640, conf=0.5)
    names = model.names
    bbox2 = prediction[0].boxes.xyxy.to('cpu').tolist()
    class_indices = []
    for r in prediction:
        for c in r.boxes.cls:
            class_indices.append(int(names[int(c)]))

    # Loop through the detected objects
    for bbox, class_idx in zip(bbox2, class_indices):
        # Check if the class is one of the classes of interest
        if class_idx in classes:
            # Convert the bounding box coordinates to integers
            bbox = [int(coord) for coord in bbox]

            # Crop the frame using the bounding box coordinates
            #cropped_frame = frame[bbox[1]-5:bbox[3]-5, bbox[0]+5:bbox[2]+5]

            # Display the cropped frame
            #cv2.imshow(f"Extracted {classes[class_idx]}", cropped_frame)

    # Display the original frame with bounding boxes
    predicted_image = prediction[0].plot()
    cv2.imshow("Original Image", predicted_image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break
    elif key & 0xFF == ord('c'):  # Press 'c' to capture the frame and perform text extraction
        # Perform text extraction using Tesseract
        for bbox, class_idx in zip(bbox2, class_indices):
            if class_idx in classes:
                bbox = [int(coord) for coord in bbox]
                cropped_frame = frame[bbox[1]-5:bbox[3]-5, bbox[0]+5:bbox[2]+5]
                text = pytesseract.image_to_string(cropped_frame)
                print(f"Extracted {classes[class_idx]}:", text)
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
