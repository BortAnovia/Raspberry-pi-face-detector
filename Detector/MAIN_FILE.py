
# coding: utf-8

# In[5]:


import cv2
#import ctypes   
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#DELETE COMMENT ON THESE TWO LINES
#from matplotlib import pyplot as plt
#from matplotlib import interactive



# NOTE: IF *LIBRARY NAME* NOT FOUND, TRY REINSTALLING IT BY GOING TO CMD AND TYPE "Pip install *LIBRARY NAME* "
#---------------------------------------------------------------------------------------------------------------------------#


# In[6]:


face_cascade = cv2.CascadeClassifier('DATA/haarcascade/haarcascade_frontalface_default.xml') 

# Import file from a location
#car_cascade = cv2.CascadeClassifier('DATA/haarcascades/vehicle_detection_haarcascades-master/cars.xml')

# NOTE, FILES DOES NOT HAVE THE SAME LOCATION IN EVERY COMPUTERS. MAKE SURE TO CHANGE THE FILE LOCATION EVERYTIME YOU
# SWITCH COMPUTERS
#---------------------------------------------------------------------------------------------------------------------------#


# In[7]:


def detector(img): 
#CRATING A NEW FUNCTION
    
    face_img = img.copy()     
    #TAKE PART OF THE IMAGE 
    face_detected = False
    face_rect = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5,minSize= (20,20)) 
    # "Face_cascade...." IS WHAT USED FOR FACIAL DETECTORS
    
    for(x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,0,255),2) 
        
        #CREATES A RECTANGLE TO SORT DETECTED FACE
        if w>0 :   
        # THE CONDITIONS ARE BASED ON THE WIDTH OF OUR RECTANGLE
            face_detected = True
    if face_detected == True : 
        
        # COUNT HOW MANY OBJECTS DETECTED BY READING THE RECTANGLE SHAPES
        cv2.putText(face_img,"Detected: "+ str(face_rect.shape[0]),(100,50),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,0,0),2)
    elif face_detected == False: 
        
        # SHOW IF NOTHING DETECTED
        cv2.putText(face_img,"Not Detected",(100,50),cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
    return face_img
#---------------------------------------------------------------------------------------------------------------------------#


# In[8]:


cap = cv2.VideoCapture(0) 
# OPENING THE CAMERA
while True:
    ret,frame = cap.read() 
    # TAKING THE "FRAME" OR PICTURES FROM CAMERA
    
    frame = detector(frame) 
    
    # APPLY THE FUNCTION THAT WE CREATED ON FRAME
    cv2.imshow('Face Detection',frame) 
    
    # SHOWING THE FRAME ON A SMALL WINDOW
    k = cv2.waitKey(1) 
    
    #THE SHUTDOWN KEY IS '27' = ESC
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
#---------------------------------------------------------------------------------------------------------------------------#

