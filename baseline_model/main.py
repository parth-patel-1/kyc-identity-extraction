from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
# Load the model
model = YOLO('aadhar_yolo_weights.pt')  # pretrained YOLOv8n model
#large_image_path = 'D:/Military_Based_XView_Dataset/tiling/test_tiled/images/1587_1280_1536.jpeg'
large_image_path = "D:/M.Tech Sem 3/NLP/Dataset/Adhar Front/adhar_ (3).jpg"

#large_image_path = "D:/Military_Based_XView_Dataset/train_images/1052.tif"

# Perform inference
results = model.track(large_image_path)
bbox = results[0].boxes
bbox2 = results[0].boxes.xyxy.to('cpu').tolist()
#prediction = model.predict(large_image_path, imgsz=640, conf=0.5)
print(bbox2)

prediction = model.predict(large_image_path, imgsz=640, conf=0.5)
predicted_image = prediction[0].plot()

predicted_image_pil = Image.fromarray(np.uint8(predicted_image))

# Display the image
predicted_image_pil.show()