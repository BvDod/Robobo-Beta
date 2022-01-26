from errno import ENETDOWN
import cv2
import imutils
import numpy as np

from PIL import Image as im


class Camera():
    def __init__(self, image_size=(128,128), debug=False, screen_segments=3):
        self.greenLower = (29, 86, 6)
        self.greenUpper = (64, 255, 255)
        self.debug = debug
        self.image_size = image_size
        self.screen_segments = screen_segments
        self.add_bottom_segment = False
        

    def getBlobs(self, image):
        """ Gets the ratio of blob to no blob and also a list which contains all blobs (location and size)"""

        image = self.resizeImage(image)
        # Pre-process image
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Detect green blobs
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        mask_ratio = np.mean(mask)/255
        
        if self.debug:
            self.showImage(image)       

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

        if self.debug:
            if len(cnts) > 0:
                cnt = max(cnts, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 5:
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
            self.showImage(blurred)
            self.showImage(mask)
            self.showImage(image)
        else:
            self.showImage(image)

        return mask_ratio, blobs
    
    def getCameraArea(self, blobs):
        """ Returns the cameraArea (1,2,3) or 0 if no blobs"""
        if not blobs:
            return 0
        area_size = 128/self.screen_segments
        camera_areas = [[i*area_size,(i+1)*area_size] for i in range(self.screen_segments)]
        print(camera_areas)
        x, y, _ = max(blobs, key=lambda blob: blob[2])

        if self.add_bottom_segment:
            if y < 128/3:
                return self.screen_segments + 1

        for i, (lower, upper) in enumerate(camera_areas):
            if x >= lower and x < upper:
                return i+1
        print("error")
        print(f"pixel = {x}")
        exit()


    def showImage(self, image):
        cv2.imshow('image',cv2.resize(image,None,fx=4, fy=4, interpolation = cv2.INTER_NEAREST))
        if self.debug == True:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
    
    def resizeImage(self, image):
        """ Resizes images to proper size """
        return cv2.resize(image, self.image_size, interpolation= cv2.INTER_LINEAR)