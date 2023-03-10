import cv2
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.extract_dataset_directory as extract

training_data_start_number = 3001
path = extract.extract_dataset_dir("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml")

#images, ids = dataCollector.getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")

def getImagesAndLabels(path, id):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # Grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
        return faceSamples, ids
    
#To-do: Load up data the way i did before instead of getImagesAndLabels
def getImagesAndLabels2(path):
    id_pathes = os.listdir(path)
    ids = []
    face_Samples = []
    for i in range(1,10178,1): ids.append(i)

    for i in range(0,len(id_pathes),1):
        tmp_pic_path_list = os.listdir(path + id_pathes[i])
        for tmp_pic_path in tmp_pic_path_list:
            PIL_img = Image.open(path + id_pathes[i] + "/" + tmp_pic_path).convert('L') # Grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for(x,y,w,h) in faces:
                face_Samples.append(img_numpy[y:y+h,x:x+w])
            status = round(float(i/len(id_pathes)*100),2)
            print("Status: " + str(status) + "%" )
    return face_Samples, ids

def getImagesAndLabels3(path):
    id_pathes = os.listdir(path)
    ids = []
    face_Samples = []
    

faces ,ids = getImagesAndLabels2(path)
print("Started training")
recognizer.train(faces,np.array(ids))
print("Training done")
recognizer.write('trainer.yml')
