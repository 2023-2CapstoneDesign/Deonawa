# Train the model
# This assumes you have the YOLOv5 repository and the necessary environment set up
import os

os.system('python yolov5/train.py --img 640 --batch 16 --epochs 30 --data data.yaml --weights yolov5s.pt')

# test
#os.system('python yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source test/images')