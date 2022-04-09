# YoloV4 on Jetson Nano 2GB or 4GB

First free some memory - use Xfce desktop manager, uninstall teamviewer. You should have 0.6-0.7GB memory consumed running the jtop only.
And you need to increase swap file to 3GB. The easiest way using jtop (preinstalled).

Install and run YoloV4

1. sudo apt update
2. sudo apt install python3-pip
3. python3 -m pip install --upgrade pip
4. pip3 install Pillow
5. pip3 install protobuf==3.5.2
6. pip3 install opencv-contrib-python==4.5.1.48 --force-reinstall
7. pip3 install numpy==1.19.4
8. clone yolov4 from https://github.com/AlexeyAB/darknet (should create darknet folder) and make
9. clone script from this repo and copy into darknet folder
10. change to darknet folder and run the script
