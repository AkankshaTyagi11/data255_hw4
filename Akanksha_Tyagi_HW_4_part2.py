import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import yaml_load, ROOT

# Run this before running the file.
# export PYTORCH_ENABLE_MPS_FALLBACK=1

data_dir = "./Squid Bat Butterfly.v2i.yolov8/valid/images"

image_paths = []
for f in os.listdir(data_dir):
    path = os.path.join(data_dir, f)
    image_paths.append(path)
print(f'Using MPS: {torch.backends.mps.is_available()}')
model = YOLO("./Squid Bat Butterfly.v2i.yolov8/runs/detect/train/weights/best.pt")
class_names = ['Bat', 'Butterfly', 'Squid']

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    results = model(img, device="mps", conf=0.9, iou=0.7)
    result = results[0]
    bboxes = result.boxes.xyxy
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for c, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(img, class_names[c], (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    cv2.imwrite(f'yolov8_infer_{idx}.png', img)
    cv2.imshow("img", img)
    cv2.waitKey(1)