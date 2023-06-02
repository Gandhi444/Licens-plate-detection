import numpy as np
import math
import cv2 as cv,cv2
def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    return 'PO12345'

def empty_callback(value):
    pass

def rect_distance(rect1,rect2):
    x1, y1, w1, h1=rect1
    x2, y2, w2, h2=rect2
    x1b=x1+w1
    y1b=y1+h1
    x2b=x2+w2
    y2b=y2+h2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return math.dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return math.dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return math.dist((x1b, y1), (x2, y2b))
    elif right and top:
        return math.dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.
    
def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 1000
    lb, ub = 0., 1.
    k=0.5
    arc_dif=np.inf
    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour
        arc_len=cv2.arcLength(contour, True)
        prev_k=k
        k = (lb + ub)/2.
        eps = k*arc_len
        approx = cv2.approxPolyDP(contour, eps, True)
        prev_arc=arc_dif
        arc_dif=abs(arc_len-cv2.arcLength(approx, True))
        print(k)
        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx
def naive_simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 1000
    
    lb, ub = 0., 1.
    k=0.5
    arc_dif=np.inf
    prev_arc_dif=np.inf
    best_k=0
    arc_len=cv2.arcLength(contour, True)
    for k in np.linspace(1,0,max_iter):
        eps = k*arc_len
        approx = cv2.approxPolyDP(contour, eps, True)
        prev_arc_dif=arc_dif
        arc_dif=abs(arc_len-cv2.arcLength(approx, True))
        if arc_dif< prev_arc_dif and len(approx)==n_corners:
            best_k=k
            prev_arc_dif=arc_dif
    return cv2.approxPolyDP(contour, best_k*arc_len, True)

def calcParams(p1,p2): #// line's equation Params computation
    if (p2[1] - p1[1] == 0):
        a = 0.0
        b = -1.0
    elif (p2[0] - p1[0] == 0):
        a = -1.0
        b = 0.0
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0

    c = (-a * p1[0]) - b * p1[1]
    return (a, b, c)

def findIntersection(params1,params2):
    x , y = -1,-1
    det = params1[0] * params2[1] - params2[0] * params1[1]
    # if (det < 0.4 and det > -0.4): # lines are approximately parallel
    #     return (-1, -1)
    #else:
    if det==0:
     return np.inf
    x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det
    y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det
    return (int(x), int(y))


def getQuadrilateral(cnt,plate,imageshape,plate_position,low1,low2,low3): 
    mask = np.zeros(plate.shape,np.uint8)
    hull=cv2.convexHull(cnt)
    mask=cv.drawContours(mask, [hull], 0, 255)
    #cv2.imshow('mask',mask)
    #lines=cv.HoughLinesP(mask, 1, np.pi / 180, 25, minLineLength=50, maxLineGap=50) #first version
    lines=cv.HoughLinesP(mask, 1, np.pi / 180, 25, minLineLength=20, maxLineGap=50)
    #lines=cv.HoughLinesP(mask, 1, np.pi / 180, low1, minLineLength=low2, maxLineGap=low3)
    cdst = cv.cvtColor(plate.copy(), cv.COLOR_GRAY2BGR)
    cdst2=cdst.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(cdst2, (l[0], l[1]), (l[2], l[3]), (255,0,255), 3, cv.LINE_AA)
        #cv.imshow('all',cdst2)
    if lines is not None and len(lines)>4:
        hor_lines=[]
        ver_lines=[]
        params=[]
        for i in range(0, len(lines)):
            l = lines[i][0]
            slope=math.degrees(math.atan2(abs(l[1]-l[3]),abs(l[0]-l[2])))
            #print(slope)
            if abs(slope)<45.0:
                hor_lines.append(l)
            else:
                ver_lines.append(l)
        first_vert=None
        second_vert=None
        first_hor=None
        second_hor=None
        for l in ver_lines:
            if first_vert is None and l[0]<plate.shape[1]/2:
                first_vert=l
            if second_vert is None and l[0]>plate.shape[1]/2:
                second_vert=l
        for l in hor_lines:
            if first_hor is None and l[1]<plate.shape[0]/2:
                first_hor=l
            if second_hor is None and l[1]>plate.shape[0]/2:
                second_hor=l
        # for line in hor_lines:
        #     cv.line(cdst, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, cv.LINE_AA)
        # for line in ver_lines:
        #     cv.line(cdst, (line[0], line[1]), (line[2], line[3]), (0,255,0), 3, cv.LINE_AA)
        lines=[]
        if first_vert is not None and second_vert is not None and first_hor is not None and second_hor is not None:
            cv.line(cdst, (first_vert[0], first_vert[1]), (first_vert[2], first_vert[3]), (255,0,255), 3, cv.LINE_AA)
            cv.line(cdst, (second_vert[0], second_vert[1]), (second_vert[2], second_vert[3]), (0,255,0), 3, cv.LINE_AA)
            cv.line(cdst, (first_hor[0], first_hor[1]), (first_hor[2], first_hor[3]), (0,0,255), 3, cv.LINE_AA)
            cv.line(cdst, (second_hor[0], second_hor[1]), (second_hor[2], second_hor[3]), (255,0,0), 3, cv.LINE_AA)
            # for i in range(0, len(lines)):
            #     l = lines[i][0]
            #     cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            lines=[first_vert,second_vert,second_hor,first_hor]

        params=[]
        if (len(lines) == 4): # we found the 4 sides
            #print("got 4 lines")
            params=[]
            for i in range(4):
                l = lines[i]
                params.append(calcParams((l[0], l[1]), (l[2], l[3])))
            corners=[]
            backup_corners=[]
            # for i in range(len(params)):
            #     for j in range(i,len(params)): #// j starts at i so we don't have duplicated points
            #         intersec = findIntersection(params[i], params[j])
            #         # if (intersec[0] > 0) and (intersec[1] > 0): #and (intersec.x < grayscale.cols) and (intersec.y < grayscale.rows):
            #         if intersec==np.inf:
            #             continue
            #         if abs(intersec[0])>imageshape[0] or abs(intersec[1])>imageshape[1]:
            #             continue
            #         if intersec[0]<0 or intersec[1]<0:
            #             backup_corners.append(intersec)
            #             continue
            #         corners.append(intersec)
            for i in range(2):
                for j in range(2):
                    intersec = findIntersection(params[i], params[2+j])
                    corners.append(intersec)
            # for i in range(len(corners)):
            #     cv.circle(cdst, corners[i], 3, (0, 255, 255))
            #cv2.imshow('lines',cdst)
            furthest_corners=[[0,0],[plate.shape[1],0],[0,plate.shape[0]],[plate.shape[1],plate.shape[0]]]
            acounted_corners=[None,None,None,None]
            #best_dist=[np.inf,np.inf,np.inf,np.inf]
            #print("good corners",corners)
            #print("backup corners",backup_corners)
            #print(backup_corners)
            if len(corners) == 4 :#// we have the 4 final corners
                print("normal")
                return corners
            else:
                for corner in corners:
                    best_fit=0
                    closet_dist=np.inf
                    for i in range(4):
                        dist=math.sqrt((corner[0]-furthest_corners[i][0])**2+(corner[1]-furthest_corners[i][1])**2)
                        if dist<closet_dist:
                            best_fit=i
                            closet_dist=dist
                    acounted_corners[best_fit]=corner
            for i,acounted in enumerate(acounted_corners):
                if acounted is None:
                    translated_corner=[furthest_corners[i][0]+plate_position[0],furthest_corners[i][1]+plate_position[1]]
                    best_fit=0
                    closet_dist=np.inf
                    for j,backup in enumerate(backup_corners):
                        if dist<closet_dist:
                            best_fit=j
                            closet_dist=dist
                        acounted_corners[i]=backup_corners[best_fit]
                else:
                    continue
            print("emergency")
            #print(acounted_corners)
            return acounted_corners
    print("somthing went wrong")
    return [[0,0],[plate.shape[1],0],[0,plate.shape[0]],[plate.shape[1],plate.shape[0]]]