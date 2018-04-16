import numpy as np
import cv2 as cv

haarpath = "../Downloads/opnecv-3.4.1/data/haarcascades/"
face_cascade = cv.CascadeClassifier(haarpath+'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(haarpath+'haarcascade_eye.xml')

cam = cv.VideoCapture(0)

while(1):

    ret,img = cam.read()
    gray = cv.cvtColor(img,0)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_area_gray = gray[y:y+h,x:x+w]
        face_area_color = img[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(face_area_gray)

        for (ex,ey,ew,eh) in eyes:

            cv.rectangle(img,(ex,ey),(ex+ew,ey+wh),(255,255,0),2)

    cv.imshow('img',img)
    k = cv.waitKey(30) & 0xff
    if k ==27:
        break

cam.release()
cv.destroyAllWindows()
