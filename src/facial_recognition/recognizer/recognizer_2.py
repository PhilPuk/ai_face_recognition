import time
start_time = time.time()
import cv2
import os
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
    def __init__(self, modelNumber, use_micro_model):
        self.trainers_path = "C:/Users/Student/Documents/models"
        self.trainer_main_name = "_percent_trainer"
        self.trainer_names = ["1","5","10"]
        self.model_number = modelNumber
        self.choosen_trainer = f"{self.trainers_path}/{self.trainer_names[self.model_number]}{self.trainer_main_name}.xml"
        if use_micro_model:
            self.choosen_trainer = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/micro_1.yml"     
        self.cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml" #cv2.data.haarcascades +
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.choosen_trainer)

model = Model(0, True)
printMessageWithTime("Model data loaded!")

class Camera():
    def __init__(self, resolution):
        self.width = resolution[0]
        self.height = resolution[1]
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3,self.width)
        self.cam.set(4,self.height)
        self.minW = 0.1*self.cam.get(3)
        self.minH = 0.1*self.cam.get(4)

Cam = Camera([640,480])
printMessageWithTime("Camera set up!")

detected_ids = []
id = 0
names = ["None", "John", "Phil", "Stew", "Stacy", "Lee"]
font = cv2.FONT_HERSHEY_TRIPLEX

while True:
    start_time = time.time()
    ret, img = Cam.cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(Cam.minW),int(Cam.minH)),
        maxSize=(400,400)
    )
    confidence = 0
    for(x,y,w,h) in faces:   
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
        id, confidence = model.recognizer.predict(gray[y:y+h,x:x+w])
        if confidence < 100:
            id = names[id] + " - " + str(id)     
        else:
            id = "unknown"
        confidence = "  {0}%".format(round(100-confidence))
        cv2.putText(img, str(id), (x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
    end_time = time.time()
    try:
        fps = 1 / (end_time - start_time)
    except:
        print("Can not calculate fps, due to division by 0!")
    print(f"[INFO] Time for detection: {end_time - start_time} - FPS: {fps}, Detected Person: {id} - Confidence: {confidence}")
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

Cam.cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Exiting program")
