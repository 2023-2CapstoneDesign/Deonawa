# Train the model
# This assumes you have the YOLOv5 repository and the necessary environment set up
import os

os.system('python yolov5/train.py --img 640 --batch 16 --epochs 10 --data data.yaml --weights yolov5s.pt')