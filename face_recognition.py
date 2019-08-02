# -*- coding: utf-8 -*-
"""
@author: DG
"""

import cv2
import os
import numpy as np

#For Detecting Face in Images
def face_find(img):
    #Give Path for haarcascade file
    face_cas = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    gray = img 
    faces = face_cas.detectMultiScale(gray,1.3,5)
    return faces,gray

#Used For Give Label to images
def label_img(filepath):
    face_list = []
    face_id=[]
    for path,subdirnames,filenames in os.walk(filepath):
        for filename in filenames:
            id=os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("path:",img_path)
            print("id:",id)
            read_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            
            faces,gray = face_find(read_img)
            if(len(faces)!=1):
                #we want to detect only one face in a image
                continue
            (x,y,w,h) = faces[0]
            roi_gray = gray[y:y+w,x:x+h] #we are cropping faces
            face_list.append(roi_gray)
            face_id.append(int(id))
    return face_list,face_id

#Training the classifier
def face_train(face_list,face_id):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(face_list,np.array(face_id))
    return model

#Predict the label of faces    
def face_recog(img_path,model):
    img= None
    test_img = None
    if type(img_path)==str:
        test_img = cv2.imread(img_path,cv2.COLOR_BGR2GRAY)   
        img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    else:
        test_img = img_path
        img = cv2.cvtColor(img_path,cv2.COLOR_BGR2GRAY)
    name = ['kriti','gal','dicaprio'] #Give the label for training in line 
    fs,g = face_find(img)
    for (x,y,w,h) in fs:
        roi_g = g[y:y+h,x:x+w]
        label,confidence = model.predict(roi_g)
        print(label)
        print(confidence)
        p_name = name[label]
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(test_img,p_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),2)    
    return test_img
    
def show_img(img):
    cv2.namedWindow('face',cv2.WINDOW_NORMAL)
    cv2.imshow('face',img) 
    k = cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
    
def test(train_images_path,test_img):
        
    #For Training run only for once
    face_list,labels = label_img(train_images_path) #training images path
    face_classifier = face_train(face_list,labels)
    face_classifier.write('face_weights_new.yml')
        
    #For Recognition
    img = face_recog(test_img,face_classifier)
    return img
    
def using_weights(test_img):
    #if you trained once and you don't want to train model again and again used this
    face_classifier = cv2.face.LBPHFaceRecognizer_create()
    
    face_classifier.read('face_weights_new.yml')
    img = face_recog(test_img,face_classifier)
    return img

if __name__=="__main__":
   final_img=test('./images','kriti.jpg')
   #first argument is path of images for training
   #second argument is path of test image 
   
   #using weights if you don't want to train again
   #final_img = using_weights('gal.jpg')
   
   #for showing image in window
   show_img(final_img)








