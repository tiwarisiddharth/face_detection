import cv2
import numpy as np

faceclass = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeclass = cv2.CascadeClassifier('haarcascade_eye.xml')

#image reading
xx=input('file name : ')
imgcam=cv2.imread(xx)
graycam=cv2.imread(xx, 0)
faces= faceclass.detectMultiScale(graycam, 1.15 ,5)
if faces is ():
    print ('Nothing found')
for (x,y,w,h) in faces:
    cv2.rectangle(imgcam, (x,y) , (x+w,y+h), (127,0,127), 2)
    cv2.imshow('Face Detected..!!', imgcam)
    cv2.waitKey(0)
    wow_gray = graycam[y:y+h , x:x+w]
    wow_color = imgcam[y:y+h , x:x+w]
    eyes=eyeclass.detectMultiScale(wow_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(wow_color, (ex,ey) , (ex+ew,ey+eh), (255,255,0) , 1)
        cv2.imshow('Face and eye', imgcam)
        cv2.waitKey(700)
cv2.destroyAllWindows()
