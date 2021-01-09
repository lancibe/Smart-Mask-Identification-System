# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:04:47 2021

@author: Lancibe
"""
import cv2

def judge(imagedir):
    imagedir='E:\\projects\\Smart_Mask_Identification_System\\data\\img\\'+imagedir
    img=cv2.imread(imagedir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray process
    
    detector= cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\haarcascade_frontalface_default.xml')
    mask_detector=cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\bin\\xml\\cascade.xml')
    faces = detector.detectMultiScale(gray, 1.01, 4)  # identify faces
    masks = mask_detector.detectMultiScale(gray, 1.01, 3)  # identify masks
    print( "faces num= " + str(len(faces)) + "masks num= " + str(len(masks)))
    
    if(len(faces) | len(masks)):
         for (x, y, w, h) in faces:
             img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw a square in face area
             img = cv2.putText(img , "nomask", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
             
         for (mx, my,mw, mh) in masks:
             img = cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)  # draw a square in mask area
             img = cv2.putText(img , "mask", (mx,my), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) # print characters
    
    
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
