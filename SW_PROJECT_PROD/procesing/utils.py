import numpy as np
import math
import cv2 as cv,cv2


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


def getQuadrilateral(cnt,plate,imageshape,plate_position): 
    mask = np.zeros(plate.shape,np.uint8)
    hull=cv2.convexHull(cnt)
    mask=cv.drawContours(mask, [hull], 0, 255)
    lines=cv.HoughLinesP(mask, 1, np.pi / 180, 25, minLineLength=20, maxLineGap=50)
    cdst = cv.cvtColor(plate.copy(), cv.COLOR_GRAY2BGR)
    cdst2=cdst.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(cdst2, (l[0], l[1]), (l[2], l[3]), (255,0,255), 3, cv.LINE_AA)
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
        lines=[]
        if first_vert is not None and second_vert is not None and first_hor is not None and second_hor is not None:
            lines=[first_vert,second_vert,second_hor,first_hor]

        params=[]
        if (len(lines) == 4): # we found the 4 sides
            params=[]
            for i in range(4):
                l = lines[i]
                params.append(calcParams((l[0], l[1]), (l[2], l[3])))
            corners=[]
            backup_corners=[]
            for i in range(2):
                for j in range(2):
                    intersec = findIntersection(params[i], params[2+j])
                    corners.append(intersec)
            for i in range(len(corners)):
                cv.circle(cdst, corners[i], 3, (0, 255, 255))
            furthest_corners=[[0,0],[plate.shape[1],0],[0,plate.shape[0]],[plate.shape[1],plate.shape[0]]]
            acounted_corners=[None,None,None,None]
            if len(corners) == 4 :#// we have the 4 final corners
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
            return acounted_corners
    return [[0,0],[plate.shape[1],0],[0,plate.shape[0]],[plate.shape[1],plate.shape[0]]]
