import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.extract_dataset_directory as extract

clear = lambda: os.system('cls')
path = extract.extract_dataset_dir("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml", "training_pre_processed")
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/trainer.yml")
detector = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml")

time_register = [1,2,3,4,5,6,7,8,9,10]
register_precision = 300 # Higher numbers = higher precision = less power

def getImagesAndLabels(path):
    id_pathes = os.listdir(path)
    chooseSizeForTraining = int(len(id_pathes) * 0.2)
    ids = []
    face_Samples = []
    id = 1
    for i in range(0,chooseSizeForTraining,1):
        tmp_pic_path_list = os.listdir(path + "/" + id_pathes[i])
        for tmp_pic_path in tmp_pic_path_list:
            PIL_img = Image.open(path + "/" + id_pathes[i] + "/" + tmp_pic_path).convert('L') # Grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for(x,y,w,h) in faces:
                face_Samples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        id += 1
        status = round(float(i/chooseSizeForTraining*100),2)
        print("Status: " + str(status) + "%" )
    return face_Samples, ids

def getImagesAndLabelsPreProcessed(path, percentageAmountOfData):
    loading_time = 0
    id_pathes = os.listdir(path)
    chooseSizeForTraining = int(len(id_pathes) * percentageAmountOfData)
    ids = []
    face_Samples = []
    id = 1
    for i in range(0,chooseSizeForTraining,1):
        start_time = time.time()
        tmp_pic_path_list = os.listdir(path + "/" + id_pathes[i])
        for tmp_pic_path in tmp_pic_path_list:
            PIL_img = Image.open(path + "/" + id_pathes[i] + "/" + tmp_pic_path)
            img_numpy = np.array(PIL_img, 'uint8')
            face_Samples.append(img_numpy)
            ids.append(id)
        id += 1
        clear()
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000,2)
        loading_time += time_ms
        time_register.insert(0,time_ms)
        time_average = round(sum(time_register) / len(time_register),2)
        time_left = round(time_average * (chooseSizeForTraining - i)/1000,2)
        if len(time_register) > register_precision:
            del time_register[-1]
        status = round(float(i/chooseSizeForTraining*100),2)
        print(f" - Status data-loading: {str(status)}%\n - Directory time: {time_ms}\n - Average Directory time: {time_average}\n - Approximately time left: {time_left}")
    time_register.clear()
    return face_Samples, ids, round(loading_time,3)

def train():
    print("Started training")
    recognizer.train(faces,np.array(ids))
    print("Training done")
    recognizer.write('trainer.yml')
    
def trainer(epochs, trainer_name):
    clear()
    total_time = 0
    print("Started training")
    for i in range(0, epochs, 1):
        start_time = time.time()
        print(f"Epoch: {i}")
        recognizer.train(faces,np.array(ids))
        recognizer.write(f'{trainer_name}.xml')
        end_time = time.time()
        time_ms = end_time - start_time
        time_register.insert(0,time_ms)
        time_average = round(sum(time_register) / len(time_register),3)
        time_left = round(time_average * (epochs - i),3)
        total_time += time_ms
        if len(time_register) > register_precision:
            del time_register[-1]
        status = round(float(i/epochs*100),2)
        print(f" - Status Training: {status}\n - Epoch time: {time_ms}\n - Average Epoch time: {time_average}\n - Approximately time left: {time_left}")
    return total_time

def settings(epochs):
    while True:
        print("Epochs amount: ",epochs)
        user_input = input("(1) Change Epoch amount (2) Go back\nInput: ")
        if user_input == "1":
            return int(input("Input new epoch amount: "))
        elif user_input == "2":
            return epochs
        else:
            print("Invalid input, pls try again!")

faces ,ids,loading_time = getImagesAndLabelsPreProcessed(path, 0.1)
epochs = 0
training_time = 0

while True:
    clear()
    print(f"[Data Loading Info] Total time: {loading_time / 1000} s")
    print(f"[Latest Training Info] Epoch amount: {epochs} Total time: {training_time / 1000}s [NOTE] Training time can be corrupted!!!")
    print("[TRAINER MENU]\n")
    print("(1) Trainer (2) Settings (3) Exit")
    user_input = input("Input: ")
    if user_input == "1":
        training_time = trainer(epochs, "10_percent_trainer")
    elif user_input == "2":
        epochs = settings(epochs)
    elif user_input == "3":
        break
    else:
        print("Invalid input, pls try again")
        time.sleep(1.5)