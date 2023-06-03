import cv2 as cv,cv2
from procesing.utils import  rect_distance,getQuadrilateral
import numpy as np

def perform_processing(img,templates):
    scale_percent = 1080/img.shape[1]
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    input = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    input_gray= cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    proccesed=input_gray
    shape=proccesed.shape
    img_center=[shape[0]/2,shape[1]/2]
    frame=0.06
    proccesed= cv2.bilateralFilter(proccesed, 13, 70, 70) 
    proccesed = cv2.Canny(proccesed,50,210)
    proccesed=cv.dilate(proccesed,np.ones((2)))

    lower_blue = np.array([95,105,125])
    upper_blue = np.array([125,255,255])
    color =input
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    mask=cv.erode(mask,np.ones((8)))
    mask=cv.dilate(mask,np.ones((10)))
    blue_cnts,new = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blue_cnts = sorted(blue_cnts, key = cv2.contourArea, reverse = True)
    img_blue_cnts=color.copy()
    
    euroband=None
    euroband_cand=[]
    for c in blue_cnts:
        x,y,w,h=cv2.boundingRect(c)
        if w/h>0.3 and w/h<0.9 and cv2.contourArea(c)<10000 and cv.contourArea(c)>200 and x<shape[0]*0.75: #and y>shape[1]*0.10:#0.3 and 1
            euroband_cand.append(c)
            euroband=euroband_cand[0]


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
    cnts,new = cv2.findContours(proccesed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    candidates=[]
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        ratio=h/w
        if ratio>0.1 and ratio<0.5 and cv2.arcLength(c,closed=True)<50000:
            candidates.append(c)


    if euroband is not None and len(candidates)>0:
        best_fit=0
        best_dist=np.inf
        euroband_rect=cv2.boundingRect(euroband)
        for i in range(len(candidates)):
            c_rect=cv2.boundingRect(candidates[i])
            dist=rect_distance(euroband_rect,c_rect)
            if dist<best_dist:
                best_dist=dist
                best_fit=i
        x,y,w,h = cv2.boundingRect(candidates[best_fit]) 
        plate_position=(x,y,w,h)
        plate=input_gray.copy()[y:y+h,x:x+w]
        plate_color=input.copy()[y:y+h,x:x+w]


    plate_processed= cv2.bilateralFilter(plate, 9, 19, 19) 
    plate_processed = cv2.Canny(plate_processed,50,210)
    plate_processed=cv.dilate(plate_processed,np.ones((2)))
    plate_processed=255-plate_processed
    cnts,h_tree = cv2.findContours(plate_processed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    candidates=[]
    candidates_hierarchy=[]
    for i,c in enumerate(cnts):
        x,y,w,h=cv2.boundingRect(c)
        ratio=h/w
        if ratio>0.1 and ratio<0.5 and cv2.arcLength(c,closed=True)<50000 and w>(plate.shape[1]*0.7) :#and len(approx) == 4:
            candidates_hierarchy.append(h_tree[0][i])
            candidates.append(c)
    lowes_level_cnt=[]
    for i,tree in enumerate(candidates_hierarchy):
        if tree[2]==-1:
            lowes_level_cnt.append(candidates[i])
    if len(lowes_level_cnt)>0:
        plate_color=cv2.drawContours(plate_color,[lowes_level_cnt[0]],-1,(255,0,0),2)
        exact_plate=lowes_level_cnt[0]

        corners=getQuadrilateral(exact_plate,plate,shape,plate_position)
        buf=False
        for corner in corners:
            if corner is None:
                buf= True
        if len(corners)==4 and not buf:
            corners_translated=[]
            for corner in corners:
                corners_translated.append([corner[0]+plate_position[0],corner[1]+plate_position[1]])
            
            sorted_corners = np.zeros((4, 2), dtype="float32")
            s = np.array(corners_translated).sum(axis=1)
            sorted_corners[0] = corners_translated[np.argmin(s)]
            sorted_corners[3] = corners_translated[np.argmax(s)]
            diff = np.diff(corners_translated, axis=1)
            sorted_corners[1] = corners_translated[np.argmin(diff)]
            sorted_corners[2] = corners_translated[np.argmax(diff)]

            pts2 = np.float32([[0,0],[plate_position[2],0],[0,plate_position[3]],[plate_position[2],plate_position[3]]])
            matrix = cv2.getPerspectiveTransform(sorted_corners, pts2)
            result = cv2.warpPerspective(input, matrix, (plate_position[2], plate_position[3]))
            

            circles_img=input.copy()
            for i in range(len(corners_translated)):
                cv.circle(circles_img, corners_translated[i], 10, (0, 0, 255))

            lower= np.array([0,20,140])
            upper= np.array([180,255,255])
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower,upper)
            result_gray=cv.cvtColor(result,cv.COLOR_BGR2GRAY)
            result_gray[mask>0]=255
            result_gray= cv2.blur(result_gray,(5,5))
            result_gray = cv2.GaussianBlur(result_gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(result_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
            _, labels = cv2.connectedComponents(thresh)
            letters = np.zeros(thresh.shape, dtype="uint8")
            total_pixels = result_gray.shape[0] * result_gray.shape[1]
            lower = total_pixels // 80 
            upper = total_pixels // 15 
            for (i, label) in enumerate(np.unique(labels)):
                if label == 0:
                    continue
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
            
                if numPixels > lower and numPixels < upper:
                    letters = cv2.add(letters, labelMask)
            cnts, _ = cv2.findContours(letters.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:7]
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            boundingBoxes=sorted(boundingBoxes,key= lambda x:x[0])
            letter_list=[]
            for box in boundingBoxes:
                x,y,w,h = box
                ratio=w/h
                if ratio>0.1 and ratio<0.9:
                    cv2.rectangle(result, (x, y), (x + w, y + h), (255,0,0), 4)
                    letter_list.append(letters[y:y+h,x:x+w])
            plate_number=""
            for letter in letter_list:
                best_result=0
                best_id=0
                for i,template in enumerate(templates):
                    temp=255-template[1][10:-20,:]
                    input = cv.resize(letter, (temp.shape[1],temp.shape[0]), interpolation = cv.INTER_AREA)
                    res =cv.matchTemplate(input,temp,cv.TM_CCOEFF_NORMED)
                    if max(res[0])>best_result:
                        best_result=max(res[0])
                        best_id=i
                plate_number=plate_number+templates[best_id][0]
            return plate_number
    return "POAAAAA"

