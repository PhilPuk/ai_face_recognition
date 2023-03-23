import time
start_time = time.time()
import cv2
import os
clear = lambda: os.system('cls')

print(cv2.__version__)
#change to opencv version 3.4.18.65

class ModelPath():
    def __init__(self, modelNumber):
        self.trainers_path = "C:/Users/Student/Documents/models"
        self.trainer_main_name = "_percent_trainer"
        self.trainer_names = ["1","5","10"]
        self.model_number = modelNumber
        self.choosen_trainer = f"{self.trainers_path}/{self.trainer_names[self.model_number]}{self.trainer_main_name}.xml"
model = ModelPath(0)

class Camera():
    def __init__(self, resolution):
        self.width = resolution[0]
        self.height = resolution[1]
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3,self.width)
        self.cam.set(4,self.height)
        self.minW = 0.25*self.cam.get(3)
        self.minH = 0.25*self.cam.get(4)
Cam = Camera([640,480])

def printMessageWithTime(message):
    print(f"{message} - Time: {time.time() - start_time}s")

printMessageWithTime("[INFO] Programm started!")
cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
printMessageWithTime("[INFO] CascadeClassifier loaded!")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model.choosen_trainer) # Choose model
printMessageWithTime("[INFO] Trainer loaded!")
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

#Init cam
cam = cv2.VideoCapture(0)
cam.set(3,640) #width
cam.set(4,480) #height

#Min window for faces
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
printMessageWithTime("[INFO] Camera set up!")

while True:
    start_time = time.time()
    clear()
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
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
    try:
        fps = 1.0 / (end_time - start_time)
    except:
        print("Can not calculate fps, due to divison by 0!")
    print(f"[INFO] Time for detection: {end_time - start_time} - FPS: {fps}, Detected Person: {id} - Confidence: {confidence}")
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Exiting program")
