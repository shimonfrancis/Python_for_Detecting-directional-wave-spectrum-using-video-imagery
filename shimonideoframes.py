# -*- coding: utf-8 -*-
import cv2
import math

videoFile = "E:/Valiathura_Camera1_Apr09-10 2017 data/170411_13/170411_13_Camera1_0002.avi"

vidcap = cv2.VideoCapture(videoFile)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("E:/Valiathura_Camera1_Apr09-10 2017 data/170411_13/images/"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1.0 #//it will capture image in each 1 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)