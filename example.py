from yolo import YOLO
from PIL import Image
import cv2

yolo = YOLO()
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
if cap.isOpened():
        while True:
                ret_val, frame = cap.read();
                cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                boxes = yolo.detect_bounding_boxes(pil_im)
                print(boxes)
else:
	print("camera open failed")
