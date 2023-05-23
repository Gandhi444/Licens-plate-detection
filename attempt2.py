import argparse
import json
from pathlib import Path
import cv2 as cv,cv2
import procesing.process
from procesing.process import empty_callback, rect_distance
import numpy as np
import os


color_imgs=[]
grey_images=[]
#folder_dir = "data/train"
folder_dir="checker/data"
for images in os.listdir(folder_dir):
    # check if the image ends with png
    if (images.endswith(".jpg") or images.endswith(".JPG")):
        input=cv.imread(folder_dir+'/'+images,cv.IMREAD_COLOR)
        scale_percent = 20 # percent of original size
        width = int(input.shape[1] * scale_percent / 100)
        height = int(input.shape[0] * scale_percent / 100)
        dim = (width, height)
        input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
        input_gray= cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        color_imgs.append(input)
        grey_images.append(input_gray)


# resize image

cv2.namedWindow('processed',)
# cv2.createTrackbar('erosion', 'processed', 1, 20000, empty_callback)
# cv2.createTrackbar('filtr', 'processed', 1, 200, empty_callback)
# cv2.createTrackbar('low', 'processed', 1, 255, empty_callback)
# cv2.createTrackbar('high', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('low1', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('low2', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('low3', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high1', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high2', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('high3', 'processed', 0, 255, empty_callback)
cv2.createTrackbar('photo nr', 'processed', 0, len(grey_images)-1, empty_callback)

while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break
    # get current positions of four trackbars
    # erosion_size = cv2.getTrackbarPos('erosion', 'processed')
    # filtr_size = cv2.getTrackbarPos('filtr', 'processed')
    # low = cv2.getTrackbarPos('low', 'processed')
    # high = cv2.getTrackbarPos('high', 'processed')
    low1 = cv2.getTrackbarPos('low1', 'processed')
    low2 = cv2.getTrackbarPos('low2', 'processed')
    low3 = cv2.getTrackbarPos('low3', 'processed')
    high1 = cv2.getTrackbarPos('high1', 'processed')
    high2 = cv2.getTrackbarPos('high2', 'processed')
    high3 = cv2.getTrackbarPos('high3', 'processed')
    nr = cv2.getTrackbarPos('photo nr', 'processed')
    proccesed=grey_images[nr]
    #proccesed=cv2.equalizeHist(proccesed)
    #proccesed= cv2.bilateralFilter(proccesed, 13, filtr_size, filtr_size) 
    
    #_,proccesed=cv2.threshold(proccesed,low,255,cv.THRESH_BINARY)
    #proccesed=cv2.GaussianBlur(proccesed,(filtr_size*2+1,filtr_size*2+1),0)
    #proccesed=cv.medianBlur(proccesed,2*filtr_size+1)
    # proccesed = cv.adaptiveThreshold(proccesed,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #          cv.THRESH_BINARY_INV,low*2+1,high)
    shape=proccesed.shape
    img_center=[shape[0]/2,shape[1]/2]
    frame=0.06
    proccesed= cv2.bilateralFilter(proccesed, 13, 70, 70) 
    proccesed = cv2.Canny(proccesed,50,210)
    proccesed=cv.dilate(proccesed,np.ones((2)))

    lower_blue = np.array([95,105,125])#100
    upper_blue = np.array([125,255,255])#120
    # lower_blue = np.array([low1,low2,low3])
    # upper_blue = np.array([high1,high2,high3])

    color =color_imgs[nr]
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    mask=cv.erode(mask,np.ones((8)))#12 previously
    mask=cv.dilate(mask,np.ones((10)))
    cv2.imshow('bluemask',mask)
    blue_cnts,new = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blue_cnts = sorted(blue_cnts, key = cv2.contourArea, reverse = True)
    # img_blue_cnts=color_imgs[nr].copy()
    img_blue_cnts=color.copy()
    
    euroband=None
    euroband_cand=[]
    for c in blue_cnts:
        x,y,w,h=cv2.boundingRect(c)
        #print(w/h)
        if w/h>0.3 and w/h<0.9 and cv2.contourArea(c)<10000 and cv.contourArea(c)>200 and x<shape[0]*0.75: #and y>shape[1]*0.10:#0.3 and 1
            #print("chosen one:",w/h)
            euroband_cand.append(c)
            euroband=euroband_cand[0]
    # best_center=np.inf
    # for cand in euroband_cand:
    #     x,y,w,h=cv.boundingRect(cand)

    img_blue_cnts=cv2.drawContours(img_blue_cnts,euroband_cand,-1,(0,255,0),2)
    proccesed[:,0:round(shape[1]*frame)]=0
    proccesed[0:round(shape[0]*frame),:]=0
    proccesed[:,shape[1]-round(shape[1]*frame):]=0
    proccesed[shape[0]-round(shape[0]*frame):,:]=0
    if euroband is not None:
        x,y,w,h=cv2.boundingRect(euroband)
        if round(x-shape[0]*frame)>round(shape[1]*frame):
            proccesed[:,0:round(x-shape[0]*frame)]=0 
        img_blue_cnts=cv2.drawContours(img_blue_cnts,[euroband],-1,(0,0,255),2)
    cv2.imshow('blue',img_blue_cnts)
    cnts,new = cv2.findContours(proccesed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    candidates=[]
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        ratio=h/w
        #print(ratio)
        if ratio>0.1 and ratio<0.5 and cv2.arcLength(c,closed=True)<50000:
            candidates.append(c)

    # img_cnts=cv2.drawContours(color_imgs[nr].copy(),cnts,-1,(0,255,0),2)
    # img_cand=cv2.drawContours(color_imgs[nr].copy(),candidates,-1,(255,0,0),2)
    # img_cand=cv2.drawContours(img_cand,[euroband],-1,(0,0,255),2)

    cv2.imshow('processed', proccesed)
    # cv2.imshow('cnt',img_cnts)
    # cv2.imshow('cand',img_cand)

    if euroband is not None and len(candidates)>0:
        best_fit=0
        best_dist=np.inf
        euroband_rect=cv2.boundingRect(euroband)
        for i in range(len(candidates)):
            c_rect=cv2.boundingRect(candidates[i])
            dist=rect_distance(euroband_rect,c_rect)
            #print(dist)
            if dist<best_dist:
                best_dist=dist
                best_fit=i
        x,y,w,h = cv2.boundingRect(candidates[best_fit]) 
        plate=grey_images[nr].copy()[y:y+h,x:x+w]
        cv2.imshow('plate',plate)
    #cv2.imshow('color',img_blue_cnts)


cv.destroyAllWindows()


