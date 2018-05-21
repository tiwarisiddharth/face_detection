import cv2
import random
import numpy as np

capture=cv2.VideoCapture(0)
ff='haarcascade_frontalface_default.xml'
facecas=cv2.CascadeClassifier(ff)
while (True):
    res,cam=capture.read()
    cam=cv2.resize(cam,(0,0),fx=1,fy=1)
    gry=cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
    faces=facecas.detectMultiScale(gry)
    for(x,y,w,h) in faces:
        cv2.rectangle(cam,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("Face It",cam)
    ch=cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
