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
cv2.createTrackbar('erosion', 'processed', 1, 10, empty_callback)
cv2.createTrackbar('filtr', 'processed', 1, 200, empty_callback)
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
    filtr_size = cv2.getTrackbarPos('filtr', 'processed')
    low = cv2.getTrackbarPos('low', 'processed')
    high = cv2.getTrackbarPos('high', 'processed')
    nr = cv2.getTrackbarPos('photo nr', 'processed')
    proccesed=grey_images[nr]

    
    # proccesed = cv.adaptiveThreshold(proccesed,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #           cv.THRESH_BINARY,11,2)
   

    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    
    proccesed= cv2.bilateralFilter(proccesed, 13, filtr_size, filtr_size) 
    proccesed = cv2.Canny(proccesed,low,high)
    proccesed=cv.dilate(proccesed,kernel)
    # proccesed=cv.dilate(proccesed,kernel)
    cnts,new = cv2.findContours(proccesed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # image1=grey_images[nr]
    # cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    cv2.imshow('processed', proccesed)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) #[:50]
    img_cnts=cv2.drawContours(color_imgs[nr].copy(),cnts,-1,(0,255,0),2)
    screenCnt = None
    candidates=[]
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) >3:
                screenCnt = approx
                x,y,w,h = cv2.boundingRect(c) 
                candidates.append(c)


    
    if len(candidates)>0:
        cnts = sorted(candidates, key = cv2.contourArea, reverse = True)
        x,y,w,h = cv2.boundingRect(cnts[0]) 
        new_img=grey_images[nr].copy()[y:y+h,x:x+w]
        image2=grey_images[nr].copy()
        screenCnt=[]
        # for cnt in cnts:
        #     perimeter = cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, 0.018 * perimeter, True) 
        #     screenCnt.append(approx)
        # cv2.drawContours(image2, screenCnt, -1, (0, 255, 0), 3)
        img_cand=cv2.drawContours(color_imgs[nr].copy(),candidates,-1,(0,255,0),2)
        cv2.imshow("image with candidates", img_cand)
        cv2.imshow("plate",new_img)

    cv2.imshow("image with contours", img_cnts)


cv.destroyAllWindows()