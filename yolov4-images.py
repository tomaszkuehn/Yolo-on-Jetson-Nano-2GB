# IP camera image capture (player) OpenCV - all in memory
# Apache 2.0 license
# Copyright (C) 2022 Tomasz Kuehn v0.2

from PIL import Image
import os
import io
import time
import requests
from requests.auth import HTTPBasicAuth
import cv2
import numpy as np
import argparse
import tempfile


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", default='/mnt/tmp/image.jpg', help="path to input image")
ap.add_argument("-y", "--yolo", default='/home/jetson/darknet/data', help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(round(time.time()))
COLORS = np.random.randint(10, 255, size=(len(LABELS), 3), dtype="uint8")


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet('/home/jetson/darknet/cfg/yolov4-tiny.cfg', '/home/jetson/darknet/yolov4-tiny.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln=[ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



fpsstart = 0
while 1:
    #from IP camera
    r = requests.get("http://192.168.0.250/cgi-bin/jpg/image.cgi", stream = False, auth = HTTPBasicAuth("Admin", "1234"))
    #from IP camera app on Android
    #r = requests.get("http://192.168.88.209:8080/shot.jpg", stream = False)
    image = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    
    points = (960, 540)
    #image = cv2.resize(image, points, interpolation= cv2.INTER_LINEAR)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []
    start = time.time()
    # loop over each of the layer outputs
    for output in layerOutputs:
	    # loop over each of the detections
	    for detection in output:
		    # extract the class ID and confidence (i.e., probability) of
		    # the current object detection
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]
		    # filter out weak predictions by ensuring the detected
		    # probability is greater than the minimum probability
		    if confidence > args["confidence"]:
			    # scale the bounding box coordinates back relative to the
			    # size of the image, keeping in mind that YOLO actually
			    # returns the center (x, y)-coordinates of the bounding
			    # box followed by the boxes' width and height
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")
			    # use the center (x, y)-coordinates to derive the top and
			    # and left corner of the bounding box
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))
			    # update our list of bounding box coordinates, confidences,
			    # and class IDs
			    boxes.append([x, y, int(width), int(height)])
			    confidences.append(float(confidence))
			    classIDs.append(classID)
    end = time.time()
    print("[INFO] Analysis took {:.6f} seconds".format(end - start))

    start = time.time()
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
	    # loop over the indexes we are keeping
	    for i in idxs.flatten():
		    # extract the bounding box coordinates
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])
		    # draw a bounding box rectangle and label on the image
		    color = [int(c) for c in COLORS[classIDs[i]]]
		    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		    text = "{}:{}: {:.2f}".format(i, LABELS[classIDs[i]], confidences[i])
		    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
	    
    diff = time.time() - fpsstart
    fpsstart = time.time()
    fps = 1.0 / diff
    color = (255, 200, 255)
    cv2.putText(image, 'FPS ' + '%.2f' % fps, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
    # show the output image
    cv2.imshow("Image", image)
    end = time.time()
    print("[INFO] DISPLAY took {:.6f} seconds".format(end - start))
    cv2.waitKey(10)




