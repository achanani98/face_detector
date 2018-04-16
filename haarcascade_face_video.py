import numpy as np
import cv2 as cv

haarpath = "../anaconda3/pkgs/opencv-3.4.1-py27_blas_openblas_200/share/OpenCV/haarcascades/"
face_cascade = cv.CascadeClassifier(haarpath+'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(haarpath+'haarcascade_eye.xml')

cam = cv.VideoCapture(0)
#cam = cv.VideoCapture('./sample.mp4')

#print cam.isOpened()
while(1):
#while(cam.isOpened()):
    ret,img = cam.read()


    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
        face_area_gray = gray[y:y+h,x:x+w]
        face_area_color = img[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(face_area_gray)

        for (ex,ey,ew,eh) in eyes:

            cv.rectangle(face_area_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),4)
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img',img)
    k = cv.waitKey(30) & 0xff
    if k ==27:
        break

cam.release()
cv.destroyAllWindows()
