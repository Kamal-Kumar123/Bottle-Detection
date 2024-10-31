# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:02:55 2024

@author: KAMAL KUMAR
"""

#now we want to detect eyes , nose, mouth means parts of mouth not the parts of whole image




#for haar cascade qualifiers first we need to videocapture by webcam(external camera) or internal camera of laptop

#face detection using opencv in python
#for face detection we have to take care for whole frame(image) b/c face may be anywhere on screen 
#region of interest :- how much region we want for working so we can also able to manage computation 
#if we want to detect the eyes, nose, mouth we have to only concentrate on face not on the whole frame as 
#region of interest    for particular feature so we have to choose our region of interest carefully

 
import cv2
#draw a boundry around the feature
#below classifier is address of classifier for which we want to make this project

def draw_boundary(img, classifier, scalefactor, minNeighbours, colorOfRectangle, textOfFeature):
    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # features = classifier.detectMultiScale(grey_img, scalefactor, minNeighbours)
   
   #or
    
    features = classifier.detectMultiScale(img, scalefactor, minNeighbours)
    #scalefactor controls the pixels of particular face suppose like a face sitting in front captures more memory
    # or pixels then a face sitting in end of room 
    
    #minNeighbours means the number of minimum feature we have to observe for trusting that particular region is face or not
    #features we want to observe in a image subregion 
     #if minimum neighbour = 3 means the next 3 features will satisfy the image condition
    
    coords = [ ]
    # x,y is the coords of rectangle top left corners
    #w , h is the width and hieght of rectangle
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),colorOfRectangle,2)   #here 2 is the thickness
        # in third parameter we have to give coordinates of bottom right corner
        #for leveling the rectangle by text we use putText    #here 0.8 is a fontFace     #here 1 is the thickness in pixels
        cv2.putText(img,textOfFeature,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,colorOfRectangle, 1,cv2.LINE_AA)
        coords = [x,y,w,h]
        
        
    return coords 
#here below faceCascade is the classifier for image
def detect(img, faceCascade,eyesCascade):
    color = {"red" : ( 0,0,255), "green" : (0,255,0),"blue":(255,0,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10 , color["blue"], "face" )
    
    if len(coords) == 4:     #suppose if there is no face detected then coords must be 0
        roi_img = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]              #understand this properly how here we are changing coordinates
        coords = draw_boundary(roi_img, eyesCascade, 1.1, 14 , color["red"], "eyes" )
    
    
    return img
    

faceCascade = cv2.CascadeClassifier(r"/home/kamal/Desktop/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier(r"/home/kamal/Desktop/haarcascade_eye.xml")


video_cap = cv2.VideoCapture(0)  #0 for our internal webcam and 1 or -1 for external webcam
while True:
    _ , img = video_cap.read()     #here we are using _ is for ignoring ret value b/c we don't use it later
    
    img =detect(img, faceCascade,eyesCascade)
    cv2.imshow("eyes_detection", img)
    if cv2.waitKey(5) == ord("q"):
        break #termination of loop
        
        
        
video_cap.release()
cv2.destroyAllWindows()