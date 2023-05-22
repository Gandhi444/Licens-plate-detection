import numpy as np
import math

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