from ultralytics import YOLO
from PIL import Image
import numpy as np
import pytesseract
from pytesseract import Output

# Load the model
model = YOLO('aadhar_yolo_weights.pt')  # pretrained YOLOv8n model
large_image_path = "C:/Users/pytho/Downloads/aadhar_test5.jpg"

# Perform inference
prediction = model.track(large_image_path, imgsz=640, conf=0.5)
names = model.names
class_indices = []
for r in prediction:
    for c in r.boxes.cls:
        print(names[int(c)])
        class_indices.append(int(names[int(c)]))

print(prediction[0].boxes)

# Define classes for AADHAAR number, date of birth, gender, name, address
classes = {0: 'Aadhar Number', 1: 'DOB', 2: 'Gender', 3: 'Name', 4: 'Address'}
bbox2 = prediction[0].boxes.xyxy.to('cpu').tolist()
print(bbox2)
print(class_indices)
# text = pytesseract.image_to_string(large_image_path)
# print(text)

for bbox, class_idx in zip(bbox2, class_indices):
    # Check if the class is one of the classes of interest
    if class_idx in classes:
        # Convert the bounding box coordinates to integers
        bbox = [int(coord) for coord in bbox]

        # Crop the image using the bounding box coordinates
        image = Image.open(large_image_path)
        cropped_img = image.crop((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5))
        cropped_img.show()

        # Use pytesseract to extract text from the cropped image
        text = pytesseract.image_to_string(cropped_img)

        # Print the extracted text
        print(f"Extracted {classes[class_idx]}:", text)

predicted_image = prediction[0].plot()

predicted_image_pil = Image.fromarray(np.uint8(predicted_image))

# Display the image
predicted_image_pil.show()

