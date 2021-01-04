# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:03:46 2021

@author: Lancibe
"""
import pandas as pd
import cv2
names=pd.read_excel('E:\\projects\\Smart_Mask_Identification_System\\data\\pos.xlsx')['names']# 读取所有照片名字
i=100000 #用于重新命名
j=0
for imagepath in names:
    #读取图片
    img = cv2.imread(imagepath)
    #转成灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #人脸识别器
    detector = cv2.CascadeClassifier('E:\\projects\\Smart_Mask_Identification_System\\haarcascade_frontalface_default.xml')
    #获取人脸位置
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        #裁减图片
        gray = gray[y:y+h,x:x+w]  # 裁剪坐标为[y0:y1, x0:x1]
        #如果人脸不为空
        try:
            # 保存裁减后的灰度图
            cv2.imwrite('E:\\projects\\Smart_Mask_Identification_System\\data\\After_Processed\\'+str(i)+'.jpg', gray)
            # cv2.waitKey(3000)
            i += 1
        except:
            print()
    j+=1       
    print(j)