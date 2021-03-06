from errno import ENETDOWN
import cv2
import imutils
import numpy as np



class Camera():
    def __init__(self, image_size=(512,512), debug=False, screen_segments=3, add_bottom_segment=False, physical=False):
        self.physical = physical
        if not physical:
            self.greenLower = (29, 86, 6)
            self.greenUpper = (64, 255, 255)
            self.redLower = (0, 86, 6)
            self.redUpper = (20, 255, 255)
        if physical:
            self.greenLower = (35, 86, 20)
            self.greenUpper = (88, 255, 255)
            self.redLower = (0, 86, 6) # TODO Tune
            self.redUpper = (20, 255, 255) # TODO Tune

        self.debug = False
        self.image_size = image_size
        self.screen_segments = screen_segments
        self.add_bottom_segment = add_bottom_segment
 

    def getBlobs(self, image, mode="green"):
        """ Gets the ratio of blob to no blob and also a list which contains all blobs (location and size)"""

        image = self.resizeImage(image)
        # Pre-process image
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Detect green blobs
        if mode == "green":
            mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        if mode == "red":
            mask = cv2.inRange(hsv, self.redLower, self.redUpper)

        self.showImage(mask)
        mask = cv2.erode(mask, None, iterations=1)
        self.showImage(mask)
        mask = cv2.dilate(mask, None, iterations=1)
        self.showImage(mask)

        mask_ratio = np.mean(mask)/255
        
        self.showImage(image, always_show=True)       
        import time
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
            self.showImage(image)
        else:
            self.showImage(image)

        return mask_ratio, blobs
    
    def getCameraArea(self, blobs):
        """ Returns the cameraArea (1,2,3) or 0 if no blobs"""
        if not blobs:
            return 0
        area_size = 512/self.screen_segments
        camera_areas = [[i*area_size,(i+1)*area_size] for i in range(self.screen_segments)]
        x, y, _ = max(blobs, key=lambda blob: blob[2])

        if self.add_bottom_segment:
            if y > (512/3) * 2:
                return self.screen_segments + 1

        for i, (lower, upper) in enumerate(camera_areas):
            if x >= lower and x < upper:
                return i+1
        print("error")
        print(f"pixel = {x}")
        exit()
    
    def hasSecuredFood(self, image):
        """ Returns the cameraArea (1,2,3) or 0 if no blobs"""
        pixels = self.image_size[0]
        gripPartImage = image[int(-(1/7)*pixels):,int((1/3)*pixels):int((2/3)*pixels)]
        self.showImage(gripPartImage, always_show=True)
        import time
        _, blobs = self.getBlobs(gripPartImage, mode="red")
        return bool(blobs)

        


    def showImage(self, image, always_show=False):
        if self.debug:
            import os
            i = 1
            while True:
                if os.path.isfile(f"images/image{i}.jpg"):
                    i += 1
                else:
                    break
            cv2.imwrite(f'images/image{i}.jpg',cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_NEAREST))

        if always_show:
            cv2.imshow("image", cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_NEAREST))
            cv2.waitKey(1)
        

    def resizeImage(self, image):
        """ Resizes images to proper size """
        return cv2.resize(image, self.image_size, interpolation= cv2.INTER_LINEAR)