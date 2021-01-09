# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:03:46 2021

@author: Lancibe
"""
import pandas as pd
import cv2
names=pd.read_excel('E:\\projects\\Smart_Mask_Identification_System\\data\\pos.xlsx')['names'] # read all names of photos
i=100000 # rename
j=0
for imagepath in names:
    # read photos
    img = cv2.imread(imagepath)
    # gray process
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # identify faces
    detector = cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\haarcascade_frontalface_default.xml')
    # obtain the coordinates
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        # cut pics
        gray = gray[y:y+h,x:x+w]  # cutting coordinates are [y0:y1, x0:x1]
        # if has face
        try:
            # save after-processing pics
            cv2.imwrite('E:\\projects\\Smart_Mask_Identification_System\\data\\After_Processed\\'+str(i)+'.jpg', gray)
            # cv2.waitKey(3000)
            i += 1
        except:
            print()
    j+=1       
    print(j)