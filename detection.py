import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA   #Principal Component Analysis

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#SVC - Support Vector Classification
#svm - Support Vector Machine

with_mask = np.load('data_with_mask.npy')
without_mask = np.load('without_mask.npy')

print(with_mask.shape)
print()
print(without_mask.shape)

with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)

print(with_mask.shape)
print()
print(without_mask.shape)

X = np.r_[with_mask , without_mask]
print(X.shape)

labels = np.zeros(X.shape[0])
labels[200:] = 1.0

names = {0:'Mask' , 1:'No Mask'}

x_train , x_test , y_train , y_test = train_test_split(X,labels ,test_size = 0.25)
print(x_train.shape)

#We have 7500 columns in x_train it is very real big problem for us. If we have lot of columns Machine Learning
#algoithm will going to slow down its speed there is technique that Dimensional Reduction in Machine Learning
#we reduce 7500 columns into 3 dimension

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

print(x_train[0] , x_train.shape)
x_train , x_test , y_train , y_test = train_test_split(X,labels ,test_size = 0.25)
svm = SVC()
svm.fit(x_train,y_train)
#x_test = pca.transform(x_test)
y_prediction = svm.predict(x_test )
accuracy_score(y_test,y_prediction)

capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag , img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h , w:w+h , :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            #face = pca.transform(face)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            print(n)

        #imS = cv2.resize(img, (1000, 1000))
        cv2.imshow('Result', img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()