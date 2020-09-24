import cv2
import sys
import numpy as np
from yolo import YOLO
from datetime import datetime
from PIL import Image
yolo = YOLO()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    ret_val, frame = cap.read();
    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    pil_image, num_boxes = yolo.detect_image(pil_im)
    
    if num_boxes > 0:
    	currDT = datetime.now()
    	pil_image.save('detected_images/detected_image_' + currDT.strftime("%H-%M-%S") + '.jpg')

    """
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    edges = open_cv_image[:, :, ::-1].copy()
    cv2.imshow('frame',edges)
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
