import pytesseract
import cv2
import os
from PIL import Image
import numpy as np
import torch
from pathlib import Path

from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="r3nfTGPGBAVKxQlVgkbV")
project = rf.workspace().project("deonawa")
model = project.version(3).model

def crop_and_save(product_name, img, box, label, save_dir='cropped_images', x_offset=0, y_offset=0):
    x1, y1, x2, y2 = map(int, box)

    # Adjust the coordinates by subtracting the offset values
    x1 -= x_offset
    x2 -= x_offset
    y1 -= y_offset
    y2 -= y_offset

    cropped_img = img[y1:y2, x1:x2]
    print(f"Image shape: {cropped_img.shape}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Extract only the filename without the path
    filename = f"{product_name}_{label}.png"
    save_path = os.path.join(save_dir, filename)
    print(f"Saving to: {save_path}")
    cv2.imwrite(save_path, cropped_img)
    print(f"Save complete for {product_name}_{label}")
    return save_path





# Function to apply Tesseract OCR on a single image
def apply_ocr(image_path):
    img = cv2.imread(image_path)
    
    # Add print statements to check image loading
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return ""

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb_image, lang='kor')
    return text


def process_directory(directory):
    for img_file in Path(directory).rglob('*.png'):
        print(f"Processing image: {img_file}")
        img = Image.open(img_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Get predictions
        predictions = model.predict(str(img_file), confidence=40, overlap=30).json()

        for prediction in predictions['predictions']:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            confidence = prediction['confidence']
            class_name = prediction['class']

            # Check if the class index is within the valid range
            label = f"{prediction['class']}"


            # Extract only the product name without any path information
            product_name = Path(img_file).stem  # This extracts the filename without extension
            # Adapt crop_and_save for Roboflow predictions
            # Adjust the offset values based on your requirements
            x_offset = 80  # Adjust as needed
            y_offset = 20  # Adjust as needed

            # Use the adjusted crop_and_save function
            cropped_image_path = crop_and_save(product_name, img_cv, (x, y, x + width, y + height), label, "cropped_images", x_offset, y_offset)
            #cropped_image_path = crop_and_save(product_name, img_cv, (x, y, x + width, y + height), label)

            ocr_text = apply_ocr(cropped_image_path)
            print(f"OCR result for {label}: {ocr_text}") 

# Replace this with your actual images directory path
images_directory = 'images'

process_directory(images_directory)