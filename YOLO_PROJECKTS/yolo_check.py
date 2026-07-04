import ultralytics
ultralytics.checks()

from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training) 

detection_outputs = model.predict(source="E:\Downloads\images (5).jpeg", conf=0.25, save=True  )  # predict on an image or video and show the results

print(detection_outputs)  # print results to the console
   
   
from PIL import Image

Image.open("C:/Users/siddharth/runs/detect/predict/images (5).jpg").show()