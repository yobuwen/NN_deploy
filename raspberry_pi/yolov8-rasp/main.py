
import yaml
from ultralytics import YOLO
import cv2 as cv

# Load a model
model = YOLO("./yolov8n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg", stream=True)  # predict on an
# print(results)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segmenation masks outputs
    probs = r.probs  # Class probabilities for classification outputs

frame = cv.imread("./bus.jpg")

# coco128.yaml target label reading.
yamlpath = "/home/yobuwen/anaconda3/envs/mytorch/lib/python3.10/site-packages/ultralytics/yolo/data/datasets/coco128.yaml"
f = open(yamlpath, 'r', encoding='utf-8')
cfg = f.read()
params = yaml.load(cfg, Loader=yaml.SafeLoader)
# print(params['names'][14])

# OPENCV Text show configuration.
fontFace = cv.FONT_HERSHEY_COMPLEX
fontScale = 0.5
fontcolor = (0, 255, 0) # BGR
thickness = 2
lineType = 4
bottomLeftOrigin = 1

# opencv to draw rectangle in image.
for ax in boxes:
    print(ax.xyxy)
    minx = int(ax.xyxy[0][0])
    miny = int(ax.xyxy[0][1])
    maxx = int(ax.xyxy[0][2])
    maxy = int(ax.xyxy[0][3])
    cv.rectangle(frame,(minx, miny),(maxx, maxy),color=(0,255,0), thickness=2)
    text = params['names'][int(ax.cls)]
    cv.putText(frame, text, (minx, miny-10), fontFace, fontScale, fontcolor, thickness, lineType)


# frame = cv.imread("./bus.jpg")
# cv.rectangle(frame,(minx, miny),(maxx, maxy),color=(0,255,0))
cv.imshow('capture',frame)
k = cv.waitKey(0)
if k == 27: # press ESC to quit.
    cv.imwrite('bus-det.jpg', frame)
    cv.destroyAllWindows()

success = model.export(format='onnx')

