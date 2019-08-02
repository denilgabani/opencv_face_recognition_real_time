# -*- coding: utf-8 -*-
"""

@author: DG
"""

'''
For Real Time Detection changes images in images folder with your own images. 
For that you can used ImagesFromVideo file to extract images from videos/webcam
and change label in name in face_recog function of face_recognition file. 
'''

import cv2
import face_recognition as fr 

camera = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
cv2.namedWindow('face',cv2.WINDOW_NORMAL)

while True:
    ret, frame = camera.read()
    final_img=fr.using_weights(frame)
    cv2.imshow('face',final_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
camera.release()
    
    
    