import cv2
import numpy as np
from PIL import Image
import os
import time
import math
clear = lambda: os.system('cls')

#Update to controll if new model should be created or existing model should be trained

#Controll panel
start_time = time.time()
time_register = [0,0]
epochs = 30
trainer_name = f"micro_1.yml"
path = "C:/Users/Student/Documents/pre_processed_micro_data_set"#extract.extract_dataset_dir("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml", "training_pre_processed")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(f"C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/{trainer_name}")

detector = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml")

def printMessageWithTime(message, start_time=0.0):
    print(f"[INFO] {message} - Time: {time.time() - start_time}s")

def getImagesAndLabelsPreProcessed(path):
    id_pathes = os.listdir(path)
    chooseSizeForTraining = int(len(id_pathes))
    ids = []
    face_Samples = []
    id = 1
    for id_path in id_pathes:
        tmp_pic_path_list = os.listdir(path + "/" + id_path)
        for tmp_pic_path in tmp_pic_path_list:
            PIL_img = Image.open(path + "/" + id_path + "/" + tmp_pic_path)
            img_numpy = np.array(PIL_img, 'uint8')
            face_Samples.append(img_numpy)
            ids.append(id)
        id += 1
    return face_Samples, ids
    
def trainer(trainer_name,faces, ids):
    recognizer.update(faces,np.array(ids))
    recognizer.write(f"{trainer_name}")
printMessageWithTime("Getting Images and Labels!", start_time)
faces ,ids = getImagesAndLabelsPreProcessed(path)
printMessageWithTime("Imaged and labels loaded - Training starting!", start_time)
total_training_time = 0.0
for i in range(epochs):
    start_time = time.time()
    trainer(trainer_name, faces, ids)
    end_time = time.time()
    printMessageWithTime(f"Epoch time: {i+1}/{epochs}", start_time=start_time)

print(f"[INFO] Training completed! - Total training time: {total_training_time}s")

