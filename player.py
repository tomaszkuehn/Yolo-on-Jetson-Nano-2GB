# IP camera image capture (player) OpenCV - all in memory
# Apache 2.0 license
# Copyright (C) 2022 Tomasz Kuehn


import os
import io
import time
import requests
from requests.auth import HTTPBasicAuth
import cv2
import numpy as np

fpsa = 0

while 1:
    start = time.time()

    r = requests.get("http://192.168.0.250/cgi-bin/jpg/image.cgi", stream = False, auth = HTTPBasicAuth("Admin", "1234"))
    #r = requests.get("http://192.168.88.209:8080/shot.jpg", stream = False)
    #image = cv2.imdecode(np.asarray(bytearray(r.content), dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    image = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    
    diff = time.time() - start

    fps = 1.0 / diff
    fpsa = (fpsa+fpsa+fps)/3
    color = (255, 0, 0)
    cv2.putText(image, 'FPS ' + '%.2f' % fpsa, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
    cv2.imshow("image", image)
    cv2.waitKey(1)
    

    




