import torch
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path

# Function to crop and save detected elements
def crop_and_save(img, box, label, save_dir='cropped_elements'):
    try:
        x1, y1, x2, y2 = map(int, box)
        cropped_img = img[y1:y2, x1:x2]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{label}.png")
        cv2.imwrite(save_path, cropped_img)
        print(f"Saved {label} element to {save_path}")
    except Exception as e:
        print(f"Error in crop_and_save: {e}")

# Function to process a directory of images
def process_directory(directory):
    try:
        # Load a pretrained YOLOv5s model from Ultralytics
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
        print(f"Loaded model: {model}")
        for img_file in Path(directory).rglob('*.png'):
            # Assuming all images are PNG
            print(f"Processing image: {img_file}")
            img = Image.open(img_file)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL to cv2 image

            # Perform inference
            results = model(img)

            # Process each detected element and save
            boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = box
                label = f"{model.names[int(cls)]}_{i}"  # Unique label
                crop_and_save(img_cv, (x1, y1, x2, y2), label)

    except Exception as e:
        print(f"Error in process_directory: {e}")

# Replace this with your actual images directory path
images_directory = '/Users/yeonu/PycharmProjects/Deonawa/images'
process_directory(images_directory)
