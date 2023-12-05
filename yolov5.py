import pytesseract
import cv2
import os
from PIL import Image
import numpy as np
import torch
from pathlib import Path


# Function to crop and save detected elements
def crop_and_save(product_name,img, box, label, save_dir='cropped_elements'):
    x1, y1, x2, y2 = map(int, box)
    cropped_img = img[y1:y2, x1:x2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{product_name}_{label}.png")
    cv2.imwrite(save_path, cropped_img)
    return save_path  # Return the path of the saved image

# Function to apply Tesseract OCR on a single image
def apply_ocr(image_path):
    img = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb_image, lang='kor+eng')
    return text

# Function to process a directory of images
def process_directory(directory):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/yeonu/PycharmProjects/Deonawa/yolov5s.pt', force_reload=True)

    for img_file in Path(directory).rglob('*.png'):  # Assuming all images are PNG
        img = Image.open(img_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = model(img)

        boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box
            label = f"{model.names[int(cls)]}_{i}"
            # Extract only the product name without the path
            product_name = str(img_file).split('/')[-1]
            product_name = product_name.replace(".png", "")
            cropped_image_path = crop_and_save(product_name,img_cv, (x1, y1, x2, y2), label)

            ocr_text = apply_ocr(cropped_image_path)
            print(f"OCR result for {label}: {ocr_text}")


# Replace this with your actual images directory path
images_directory = '/Users/yeonu/PycharmProjects/Deonawa/images'
process_directory(images_directory)
