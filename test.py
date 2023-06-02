import argparse
import json
from pathlib import Path
import cv2 as cv,cv2
import procesing.process
from procesing.process import empty_callback, rect_distance
import numpy as np
import os

# resize image

cv2.namedWindow('processed',)
cv2.createTrackbar('low1', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('low2', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('low3', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high1', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high2', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high3', 'processed', 0, 255, empty_callback)

#input=cv.imread("colors.jpg",cv.IMREAD_COLOR)


input=cv.imread("colors.jpg",cv.IMREAD_COLOR)
# scale_percent = 20 # percent of original size
# width = int(input.shape[1] * scale_percent / 100)
# height = int(input.shape[0] * scale_percent / 100)
# dim = (width, height)
# input = cv.resize(input, dim, interpolation = cv.INTER_AREA)


while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
    # escape key pressed
        break
    processed=input
    low1 = cv2.getTrackbarPos('low1', 'processed')
    low2 = cv2.getTrackbarPos('low2', 'processed')
    low3 = cv2.getTrackbarPos('low3', 'processed')
    high1 = cv2.getTrackbarPos('high1', 'processed')
    high2 = cv2.getTrackbarPos('high2', 'processed')
    high3 = cv2.getTrackbarPos('high3', 'processed')


    lower_blue = np.array([low1,low2,low3])
    upper_blue = np.array([high1,high2,high3])
    # lower_blue = np.array([100,80,80])
    # upper_blue = np.array([120,255,255])
    print(lower_blue)
    hsv = cv2.cvtColor(input.copy(), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    cv2.imshow('mask',mask)
    processed=cv2.bitwise_or(input.copy(),input.copy(),mask=mask)
    cv2.imshow('processed',processed)
cv.destroyAllWindows()
