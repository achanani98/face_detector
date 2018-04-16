import numpy as np
import cv2 as cv
import os
r = cv.face.LBPHFaceRecognizer_create()

folders = os.listdir("./dataset/")
images = []
labels = []
label_key = 0
label_keys = []

for folder in folders:
    label = folder
    files = os.listdir("./dataset/"+str(folder)+"/")
    for file in files:
        img = cv.imread("dataset/"+str(folder)+"/"+str(file),0)
        images.append(np.array(img))
        label_keys.append(label_key)
    label_key += 1
    labels.append(str(label))


labels = np.array(labels)
images = np.array(images)
label_keys = np.array(label_keys)

r.train(images,label_keys)


haarpath = "../anaconda3/pkgs/opencv-3.4.1-py27_blas_openblas_200/share/OpenCV/haarcascades/"
face_cascade = cv.CascadeClassifier(haarpath+"haarcascade_frontalface_default.xml")

#img = cv.imread("./is3.jpg",1)

#cam = cv.VideoCapture(0)
cam = cv.VideoCapture('sample.mp4')
#while(1):
while(cam.isOpened()):

    ret,img = cam.read()

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    font = cv.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in faces:
        face_area_gray = gray[y:y+h,x:x+w]
        face_area_color = img[y:y+h,x:x+w]
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
        cv.putText(img,labels[r.predict(face_area_gray)[0]], (x,y), font, 2, (255, 255, 255), 5, cv.LINE_AA)
        #print labels[r.predict(face_area_gray)[0]]
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img',img)
    k = cv.waitKey(30) & 0xff
    if k ==27:
        break
cam.release()
cv.destroyAllWindows()
