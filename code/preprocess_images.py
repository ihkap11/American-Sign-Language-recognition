
# coding: utf-8

# In[ ]:

import cv2
import numpy as np

dim = 50

def resize_image(img, size = dim):
    """
    test and train images by default are taken of dimension 100 x 100
    """
    
    img = cv2.resize(img, (size, size))
    return img




def make_skin_white_rest_black(img):
    """
    Extracts skin area (hand) from the frame
    """   
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur to help remove noise
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # tuned to detect skin color
    lower_color = np.array([3, 50, 50])
    upper_color = np.array([33,255,255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(mask, kernel, iterations = 2)
    skinMask = cv2.dilate(mask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(mask, (3, 3), 0)
    

    result = cv2.bitwise_and(img, img, mask = skinMask)     
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    
    ret, thresh = cv2.threshold(result, 75,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
#     cv2.convertScaleAbs(closing, closing)
    
    gray1 = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    
    return gray1
    
    
    
    
    
def apply_preprocessing(img):    
    img = make_skin_white_rest_black(img)
    img = resize_image(img, dim)
    return img

