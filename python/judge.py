# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:04:47 2021

@author: Lancibe
"""
import cv2

def judge(imagedir):
    imagedir='E:\\projects\\Smart_Mask_Identification_System\\data\\img\\'+imagedir
    img=cv2.imread(imagedir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图像灰化
    
    detector= cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\haarcascade_frontalface_default.xml')
    mask_detector=cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\bin\\xml\\cascade.xml')
    faces = detector.detectMultiScale(gray, 1.01, 4)  # 识别人脸  1.02  每次检测的放大倍率与被检查物体成像大小相关
    masks = mask_detector.detectMultiScale(gray, 1.01, 3)  # 识别人脸  1.02  )
    print( "faces num= " + str(len(faces)) + "masks num= " + str(len(masks)))
    
    if(len(faces) | len(masks)):
         for (x, y, w, h) in faces:
             img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在人脸区域画一个正方形出来
             img = cv2.putText(img , "nomask", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
             
         for (mx, my,mw, mh) in masks:
             img = cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)  # 在戴口罩人脸区域画一个正方形出来
             img = cv2.putText(img , "mask", (mx,my), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) #在戴口罩人脸区域输出字符
    
    
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
