import torch
from PIL import Image
import numpy as np
import cv2
import os
from ultralytics import YOLO

# Function to crop and save detected elements
def crop_and_save(img, box, label, save_dir='cropped_elements'):
    x1, y1, x2, y2 = map(int, box)
    cropped_img = img[y1:y2, x1:x2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{label}.png")
    cv2.imwrite(save_path, cropped_img)
    print(f"Saved {label} element to {save_path}")

# Load the model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = YOLO('yolov5s.pt')
# Load an image
img_path = '/Users/yeonu/PycharmProjects/Deonawa/images/Apple 아이패드 프로 11 4세대 M2 512GB 실버 MNXJ3KH.png'
img = Image.open(img_path)
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL Image to cv2 image

# Perform inference
results = model(img)

# Results
results.print()  # Print results to console
results.show()  # Show results

# Process each detected element and save the cropped images
boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
for i, box in enumerate(boxes):
    x1, y1, x2, y2, conf, cls = box
    label = f"{model.names[int(cls)]}_{i}"  # Unique label for each element
    crop_and_save(img_cv, (x1, y1, x2, y2), label)
