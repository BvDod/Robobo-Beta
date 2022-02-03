import cv2
import imutils
import numpy as np



def getBlobs(image):
    """ Gets the ratio of blob to no blob and also a list which contains all blobs (location and size)"""

    greenLower = (35, 86, 20)
    greenUpper = (88, 255, 255)
    redLower = (0, 86, 6) # TODO Tune
    redUpper = (20, 255, 255) # TODO Tune

    image = resizeImage(image)
    # Pre-process image
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Detect green blobs
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    mask_ratio = np.mean(mask)/255
    
    
    showImage(image)       

    # Find contours of blobs
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    blobs =[]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 1:
            x_center, y_center = (x+(w/2)), (y+(h/2))
            area = w*h
            blobs.append([x_center, y_center, area])
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 0, 255), 1)

   
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 5:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
    showImage(blurred)
    showImage(mask)
    showImage(image)
    

    return mask_ratio, blobs

def showImage(image):
    cv2.imshow('image',cv2.resize(image,None,fx=4, fy=4, interpolation = cv2.INTER_NEAREST))
    cv2.waitKey(0)
    

def resizeImage(image):
    """ Resizes images to proper size """
    return cv2.resize(image, (128,128), interpolation= cv2.INTER_LINEAR)


import os

for filename in os.listdir('test_images/'):    
    image = cv2.imread(f'test_images/{filename}')
    getBlobs(image)