# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:50:01 2019

@author: DG
"""


import cv2

def face_find(img):
    #Give Path for haarcascade file
    face_cas = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray,1.3,5)
    if len(faces)==1:
        return 1
    else:
        0


camera = cv2.VideoCapture(0)
path = "./bhautik/"
count=1
while (True):
    
    ret,frame = camera.read()
    if ret:
        
        cv2.imshow('img',cv2.resize(frame,(500,500)))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        flag = face_find(frame) 
        if(flag):
            filename = path+str(count)+'.jpg'
            cv2.imwrite(filename,frame)
            print(count)
    
            count+=1
        
cv2.destroyAllWindows()
camera.close()







