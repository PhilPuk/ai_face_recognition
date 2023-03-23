import time
start_time = time.time()
import cv2
import os
import numpy as np
clear = lambda: os.system('cls')

def printMessageWithTime(message):
    print(f"[INFO] {message} - Time: {time.time() - start_time}s")
    
def most_frequent_element_in_list(list):
    counter = 0
    num = list[0]
    
    for i in list:
        curr_frequency = list.count(i)
        if(curr_frequency>counter):
            counter = curr_frequency
            num = i
    return num

printMessageWithTime("Programm started, packages loaded!")
print("OpenCV Version: " + cv2.__version__)

class Model():
    def __init__(self, modelNumber):
        self.trainers_path = "C:/Users/Student/Documents/models"
        self.trainer_main_name = "_percent_trainer"
        self.trainer_names = ["1","5","10"]
        self.model_number = modelNumber
        self.choosen_trainer = f"{self.trainers_path}/{self.trainer_names[self.model_number]}{self.trainer_main_name}.xml"
        self.cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml" #cv2.data.haarcascades +
        #self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.classifier_cuda = cv2.cuda_CascadeClassifier(self.cascadePath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.choosen_trainer)

model = Model(0)
printMessageWithTime("Model data loaded!")

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
printMessageWithTime("Camera set up!")

detected_ids = []
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

printMessageWithTime("Program loop started")
while True:
    start_time = time.time()
    ret, frame = Cam.cam.read()
    if ret:
        printMessageWithTime("Captured frame!")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Need gpu way to convert to grayscale
        cuFrame = cv2.cuda_GpuMat(frame)
        mat = cv2.cuda_GpuMat()
        printMessageWithTime("Uploaded frame to gpu!")
        try:
            result = model.classifier_cuda.detectMultiScale(cuFrame, mat)
            print("Copied detecter!")
            gpu_result = result.download()
            printMessageWithTime("Downloaded frame from gpu!")
            print(result)
        except:
            printMessageWithTime("Could not download frame from gpu!")
        id = 0
        confidence = 0
        printMessageWithTime("Starting Prediction!")
        for(x,y,w,h) in gpu_result:
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 2)
            id, confidence = model.recognizer.predict(gpu_result[y:y+h,x:x+w])
            if confidence < 100:
                id = str(id)            
            else:
                id = "unknown"
            confidence = "  {0}%".format(round(100-confidence))
            cv2.putText(frame, str(id), (x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(frame,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
        end_time = time.time()
        detected_ids.insert(0,id)
        if len(detected_ids) > 40:
            del detected_ids[0]
        most_frequent_id = most_frequent_element_in_list(detected_ids)
        print(f"[INFO] Time for detection: {end_time - start_time}, Detected Person: {most_frequent_id} - Confidence: {confidence}")
        cv2.imshow('camera',frame)
    else:
        printMessageWithTime("Continue executed")
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Cam.cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Exiting program")
