import argparse
import json
from pathlib import Path
import cv2 as cv,cv2
import procesing.process
from procesing.process import empty_callback, rect_distance ,simplify_contour,naive_simplify_contour,getQuadrilateral
import numpy as np
import os


color_imgs=[]
grey_images=[]
folder_dir = "data/train"
#folder_dir="checker/data"
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
    #cv2.imshow('bluemask',mask)
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
    #cv2.imshow('blue',img_blue_cnts)
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

    # cv2.imshow('processed', proccesed)
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
        plate_position=(x,y,w,h)
        plate=grey_images[nr].copy()[y:y+h,x:x+w]
        plate_color=color_imgs[nr].copy()[y:y+h,x:x+w]
        #cv2.imshow('plate',plate)
    #cv2.imshow('color',img_blue_cnts)


    plate_processed= cv2.bilateralFilter(plate, 9, 19, 19) 
    plate_processed = cv2.Canny(plate_processed,50,210)
    plate_processed=cv.dilate(plate_processed,np.ones((2)))
    plate_processed=255-plate_processed
    #mask,plate_processed=cv2.threshold(plate,low1,255,cv.THRESH_BINARY)
    # plate = cv.adaptiveThreshold(plate,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #           cv.THRESH_BINARY,11,2)
    cnts,h_tree = cv2.findContours(plate_processed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #plate_all_cnts=cv2.drawContours(plate_color,cnts,-1,(0,0,255),2)
    #cnts=
   # print(h_tree)
    #print(len(lowes_level_cnt))
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    candidates=[]
    candidates_hierarchy=[]
    for i,c in enumerate(cnts):
        x,y,w,h=cv2.boundingRect(c)
        # peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.001*peri, True)
        # print(len(approx))
        ratio=h/w
        #print(ratio)
        if ratio>0.1 and ratio<0.5 and cv2.arcLength(c,closed=True)<50000 and w>(plate.shape[1]*0.7) :#and len(approx) == 4:
            # peri = cv2.arcLength(c, True)
            # approx = cv2.approxPolyDP(c, 0.8*peri, True)
            # print(len(approx))
            # print(w,plate.shape)
            candidates_hierarchy.append(h_tree[0][i])
            candidates.append(c)
    lowes_level_cnt=[]
    for i,tree in enumerate(candidates_hierarchy):
        if tree[2]==-1:
            lowes_level_cnt.append(candidates[i])
    if len(lowes_level_cnt)>0:
        plate_color=cv2.drawContours(plate_color,[lowes_level_cnt[0]],-1,(255,0,0),2)
        exact_plate=lowes_level_cnt[0]
        # peri = cv2.arcLength(exact_plate, True)
        # approx = cv2.approxPolyDP(exact_plate, 0.05*peri, True)
        #approx=naive_simplify_contour(exact_plate,4)
        corners=getQuadrilateral(exact_plate,plate,shape,plate_position,low1,low2,low3)
        #print(corners)
        buf=False
        for corner in corners:
            #print(corner)
            if corner is None:
                buf= True
            #print(buf)
        if len(corners)==4 and not buf:
            corners_translated=[]
            for corner in corners:
                corners_translated.append([corner[0]+plate_position[0],corner[1]+plate_position[1]])
            #print('translated',corners_translated)
            
            sorted_corners = np.zeros((4, 2), dtype="float32")
            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = np.array(corners_translated).sum(axis=1)
            sorted_corners[0] = corners_translated[np.argmin(s)]
            sorted_corners[3] = corners_translated[np.argmax(s)]
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(corners_translated, axis=1)
            sorted_corners[1] = corners_translated[np.argmin(diff)]
            sorted_corners[2] = corners_translated[np.argmax(diff)]

            pts2 = np.float32([[0,0],[plate_position[2],0],[0,plate_position[3]],[plate_position[2],plate_position[3]]])
            matrix = cv2.getPerspectiveTransform(sorted_corners, pts2)
            result = cv2.warpPerspective(color_imgs[nr], matrix, (plate_position[2], plate_position[3]))
            

            circles_img=color_imgs[nr].copy()
            for i in range(len(corners_translated)):
                cv.circle(circles_img, corners_translated[i], 10, (0, 0, 255))
            #cv2.imshow('circles',circles_img)

            # result_hsv=cv.cvtColor(result,cv.COLOR_BGR2HSV)
            # # result_hsv = cv2.inRange(result_hsv.copy(), (180,low1,low2), (180, high1+1, high2+1))
            # # result_hsv=cv.cvtColor(result_hsv,cv.COLOR_HSV2BGR)
            # lwr = np.array([0, 0, 0])
            # upr = np.array([180, 255, low1])
            # msk = cv2.inRange(result_hsv, lwr, upr)
            # cv.imshow('hsv',result_hsv)
           # res=cv.bitwise_and(msk,result_hsv)
            lower= np.array([0,20,140])#100
            upper= np.array([180,255,255])#120
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower,upper)
            result_gray=cv.cvtColor(result,cv.COLOR_BGR2GRAY)
            result_gray[mask>0]=255
            result_gray= cv2.blur(result_gray,(5,5))
            result_gray = cv2.GaussianBlur(result_gray, (5, 5), 0)
            #cv.imshow('test',result_gray)
            thresh = cv2.adaptiveThreshold(result_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
            _, labels = cv2.connectedComponents(thresh)
            letters = np.zeros(thresh.shape, dtype="uint8")
            total_pixels = result_gray.shape[0] * result_gray.shape[1]
            lower = total_pixels // 80 #(low1+2) # heuristic param, can be fine tuned if necessary
            upper = total_pixels // 15 #(low2+2)
            for (i, label) in enumerate(np.unique(labels)):
                # If this is the background label, ignore it
                if label == 0:
                    continue
            
                # Otherwise, construct the label mask to display only connected component
                # for the current label
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
            
                # If the number of pixels in the component is between lower bound and upper bound, 
                # add it to our mask
                if numPixels > lower and numPixels < upper:
                    letters = cv2.add(letters, labelMask)
                cnts, _ = cv2.findContours(letters.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:7]
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                for box in boundingBoxes:
                    x,y,w,h = box
                    ratio=w/h
                    if ratio>0.1 and ratio<0.9:
                        cv2.rectangle(result, (x, y), (x + w, y + h), (255,0,0), 4)
            cv2.imshow('gray_res',letters)
            cv2.imshow('res',result)





    cv2.imshow('processed', plate_processed)
    


    # plate_hsv = cv2.cvtColor(plate_color, cv2.COLOR_BGR2HSV)
    # mask=cv.inRange(plate_hsv, (0,low1,low2), (180,255,255))
    # black_mask=[180,low2,low3]
    # plate_hsv[plate_hsv<black_mask]=[0,0,255]
    # plate_hsv = cv2.cvtColor(plate_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('plate',plate_color)
    #cv2.imshow('plate cnts',plate_all_cnts)
cv.destroyAllWindows()


