import os
import cv2 as cv
#path yo to positive pics
path = '../Downloads/Selfie/'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    img = cv.imread(path + str(i)+'.jpg',cv.IMREAD_COLOR)
    img = cv.resize(img,(360,640))
    cv.imwrite(path + '_1_'+str(i)+'.jpg',img)
    i = i+1
