#mount -t tmpfs -o size=50M tmpfs /mnt/tmp

from PIL import Image
import pygame
import os
import io
import time
import requests
from requests.auth import HTTPBasicAuth
import cv2
import numpy as np
import argparse
import tempfile

def download_file(user_name, user_pwd, url, file_name):
    #file_name = url.rsplit('/', 1)[-1]
    with requests.get(url, stream = True, auth = HTTPBasicAuth(user_name, user_pwd)) as response:
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", default='/mnt/tmp/image.jpg', help="path to input image")
ap.add_argument("-y", "--yolo", default='/home/tomasz/darknet/data', help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(round(time.time()))
COLORS = np.random.randint(10, 255, size=(len(LABELS), 3), dtype="uint8")

#"""
#initialize display window PYGAME
pygame.init()
displayWidth = 1920
displayHeight = 540
surface = pygame.display.set_mode((displayWidth, displayHeight))
pygame.display.set_caption('Pygame image')
surface.fill((100,100,100))
pygame.display.update()
#--pygame
#"""

#darknet_path = '/home/tomasz/darknet/'

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet('/home/tomasz/darknet/cfg/yolov4.cfg', '/home/tomasz/darknet/yolov4.weights')
ln = net.getLayerNames()
ln=[ln[i - 1] for i in net.getUnconnectedOutLayers()]

counter = 0
while 1:
    counter = counter + 1
    download_file("Admin", "1234", "http://192.168.0.250/cgi-bin/jpg/image.cgi", "/mnt/tmp/image.jpg")
    
    #read to variable
    buffer = tempfile.SpooledTemporaryFile(150000)
    with requests.get("http://192.168.0.250/cgi-bin/jpg/image.cgi", stream = True, auth = HTTPBasicAuth("Admin", "1234")) as response:
        for chunk in response.iter_content(chunk_size = 8192):
            buffer.write(chunk)
    buffer.seek(0)
    im = Image.open(io.BytesIO(buffer.read()))
    buffer.close()
    mode = im.mode
    size = im.size
    data = im.tobytes()
    py_image = pygame.image.fromstring(data, size, mode)
    surface.blit(py_image, (0, 0))
    pygame.display.update()

    image = cv2.imread('/mnt/tmp/image.jpg')
    points = (480, 270)
    #image = cv2.resize(image, points, interpolation= cv2.INTER_LINEAR)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #cv2.imshow("image", image)
    #cv2.waitKey(100)
    
    """
    #pygame image
    #im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #im = np.fliplr(im)
    #im = np.rot90(im)
    pyimg = pygame.surfarray.make_surface(im)
    pyimg = pygame.transform.scale(pyimg, (960, 540)) 

    surface.blit(pyimg, (0, 0))
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render('Count '+f'{counter}', True, (0,0,0), (200,0,0))
    textRect = text.get_rect()
    surface.blit(text, textRect)
    pygame.display.update()
    #--pygame
    """ 
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    #print(layerOutputs)

    boxes = []
    confidences = []
    classIDs = []
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
		    if confidence > -1: #args["confidence"]:
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
    cv2.putText(image, 'COUNT ' + f'{counter}', (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(1)




time.sleep(3)

