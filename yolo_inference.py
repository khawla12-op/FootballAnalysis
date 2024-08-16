# from ultralytics import YOLO
from yolov9 import YOLOv9
# Load the YOLO model
model = YOLOv9('models/best.pt')
results = model.predict('Input_videos/testVideo.mp4', save=True, stream=True)
results_list = list(results)
print(results_list[0])
# Iterate through the bounding boxes in the first result
for box in results_list[0].boxes:
    print(box)
