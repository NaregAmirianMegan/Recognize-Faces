'''
@author: Nareg A. Megan
'''

import cv2
import numpy as np
import os

recognizer = cv2.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_data(id, faces, labels):
    cam = cv2.VideoCapture(0)
    sampleNum = 0

    while(True):
        ret,img = cam.read()
        #convert image to grayscale
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #returns location and width, height of detected face
        face = detector.detectMultiScale(grayImg,1.3,5)

        #for each detection create and save grayscaled image to data/User. format
        for(x,y,w,h) in face:
            sampleNum = sampleNum + 1
            #create image
            faces.append(grayImg[y:y+h,x:x+w])
            labels.append(id)
            #draw rectangle around detected face
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face",img)
        cv2.waitKey(1)
        #collect 20 samples
        if(sampleNum == 100):
            break
    #close webcam and cv2
    cam.release()
    cv2.destroyAllWindows()


id = ""
count = 1
faces = []
labels = []
subjects = []

while(True):
    id = raw_input('Enter user ID: ')
    if(id == "q"):
        break
    create_data(count, faces, labels)
    subjects.append(id)
    count += 1

#testing purposes
print("faces: ", faces)
print("labels: ", labels)

#train local binary pattern historgram model on faces and labels
#created previously
recognizer.train(faces, np.array(labels))
cam = cv2.VideoCapture(0)

#detect faces on screen
while(True):
    ret,img = cam.read()
    #convert image to grayscale
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #returns location and width, height of detected face
    face = detector.detectMultiScale(grayImg,1.3,5)

    #for each detection create and save grayscaled image to data/User. format
    for(x,y,w,h) in face:
        label = recognizer.predict(grayImg[y:y+h,x:x+w])
        print(label[0])
        label_text = subjects[int(label[0])-1]
        #draw rectangle around detected face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face",img)
    if(cv2.waitKey(10) == ord('q')):
        break

#close webcam and cv2
cam.release()
cv2.destroyAllWindows()
