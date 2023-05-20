import argparse
import json
from pathlib import Path
import cv2 as cv,cv2
import procesing.process
from procesing.process import empty_callback
import numpy as np
import os

color_imgs=[]
grey_images=[]
folder_dir = "data/train"
for images in os.listdir(folder_dir):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        input=cv.imread(folder_dir+'/'+images,cv.IMREAD_COLOR)
        scale_percent = 10 # percent of original size
        width = int(input.shape[1] * scale_percent / 100)
        height = int(input.shape[0] * scale_percent / 100)
        dim = (width, height)
        input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
        input_gray= cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        color_imgs.append(input)
        grey_images.append(input_gray)


# resize image

# input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
# input_gray= cv.cvtColor(input, cv.COLOR_BGR2GRAY)
# input=cv.imread('data/train/2023-05-08 (1).jpg',cv.IMREAD_COLOR)
# scale_percent = 10 # percent of original size
# width = int(input.shape[1] * scale_percent / 100)
# height = int(input.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image

# input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
# input_gray= cv.cvtColor(input, cv.COLOR_BGR2GRAY)
cv2.namedWindow('processed',)
cv2.createTrackbar('erosion', 'processed', 1, 100, empty_callback)
cv2.createTrackbar('low', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('photo nr', 'processed', 0, len(grey_images)-1, empty_callback)

while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break
    # get current positions of four trackbars
    erosion_size = cv2.getTrackbarPos('erosion', 'processed')
    low = cv2.getTrackbarPos('low', 'processed')
    high = cv2.getTrackbarPos('high', 'processed')
    nr = cv2.getTrackbarPos('photo nr', 'processed')
    proccesed=grey_images[nr]

    #proccesed=cv.GaussianBlur(proccesed,(9,9),0)
    proccesed= cv2.bilateralFilter(proccesed, 11, low, low) 
    proccesed = cv.adaptiveThreshold(proccesed,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv.THRESH_BINARY,11,2)
    kernel = np.ones((high, high), np.uint8)
    proccesed=cv.dilate(proccesed,kernel)


    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    proccesed=cv.erode(proccesed,kernel)


    cv2.imshow('processed', proccesed)




cv.destroyAllWindows()