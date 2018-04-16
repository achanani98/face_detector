import cv2 as cv
import numpy as np

haarpath = "../anaconda3/pkgs/opencv-3.4.1-py27_blas_openblas_200/share/OpenCV/haarcascades/"
face_cascade = cv.CascadeClassifier(haarpath+"haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(haarpath+"haarcascade_eye.xml")


#img = cv.imread("../dataset/selfies/194.jpg",1)
#img = cv.imread('is1.jpg')

img = cv.imread('abc.jpg',1)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:

    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    face_area_gray = gray[y:y+h,x:x+w]
    face_area_color = img[y:y+h,x:x+w]

    eyes = eye_cascade.detectMultiScale(face_area_gray)

    for (ex,ey,ew,eh) in eyes:

        cv.rectangle(face_area_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),5)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
