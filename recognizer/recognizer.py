import time
start_time = time.time()
import cv2
import numpy as np
import os

trainer_paths = [
    "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/10_percent_trainer.xml",
    "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/5_percent_trainer.yml",
    "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/1_percent_trainer.yml",
    "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/trainer.yml"
]
model_number = 0

print(f"[INFO] Programm started! - Time: {time.time() - start_time}s")
cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
print(f"[INFO] CascadeClassifier loaded! - Time: {time.time() - start_time}s")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_paths[model_number]) # Choose model
print(f"[INFO] Trainer loaded! - Time: {time.time() - start_time}s")
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

names = {"None" : 0, "Phil" : 10178}

#Init cam
cam = cv2.VideoCapture(0)
cam.set(3,640) #width
cam.set(4,480) #height

#Min window for faces
minW = 0.25*cam.get(3)
minH = 0.25*cam.get(4)
print(f"[INFO] Camera set up! - Time: {time.time() - start_time}s")
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW),int(minH)),
        maxSize=(400,400)
    )
    id = 0
    confidence = 0
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence < 100:
            id = str(id)            
        else:
            id = "unknown"
        confidence = "  {0}%".format(round(100-confidence))
        cv2.putText(img, str(id), (x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
    end_time = time.time()
    print(f"[INFO] Time for detection: {end_time - start_time}, Detected Person: {id} - Confidence: {confidence}")
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Exiting program")
