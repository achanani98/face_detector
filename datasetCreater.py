
import os
import cv2 as cv
import numpy as np
import copy


sample_path = "Camera/"
files = os.listdir(sample_path)
storage_path = "positive/"

haarpath = "../anaconda3/pkgs/opencv-3.4.1-py27_blas_openblas_200/share/OpenCV/haarcascades/"
face_cascade = cv.CascadeClassifier(haarpath+"haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(haarpath+"haarcascade_eye.xml")



i = 1
j = 1

for file in files:

    print file
    img = cv.imread(sample_path + file)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

            face_area_color = copy.deepcopy(img[y:y+h,x:x+w])


            face_area_color_resized = cv.resize(face_area_color,(90,160))
            if cv.imwrite(storage_path+str(i)+".jpg",face_area_color_resized):
                print "file no :" + str(j) + "  face no :"+ str(i) + " saved succesfully"
                i += 1
            else:
                print "dayummmm girl u an ugly imageeeeeeeeeeeeeee"

    j += 1
    print "image number: " + str(j)
