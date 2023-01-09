import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('E:\\IMG_1945-1.jpg')
# cv2.imshow('IMg',img)
# cv2.waitKey(5)

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print(haar_data.detectMultiScale(img))

#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)

# while True:
#     faces = haar_data.detectMultiScale(img)
#     for x,y,w,h in faces:
#         cv2.rectangle(img , (x,y) , (w+x , h+y),(255,0,255),4)
#     imS = cv2.resize(img,(1000,1000))
#     cv2.imshow('Result', imS)
#     if cv2.waitKey(2) == 27:
#         break
# cv2.destroyAllWindows()

capture = cv2.VideoCapture(0)
data = []
while True:
    flag , img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h , w:w+h , :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data) < 200:
                data.append(face)
        #imS = cv2.resize(img, (1000, 1000))
        cv2.imshow('Result', img)
        if cv2.waitKey(2) == 27 or len(data) >=200:
            break
capture.release()
cv2.destroyAllWindows()