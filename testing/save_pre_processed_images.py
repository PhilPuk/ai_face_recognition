import cv2
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.extract_dataset_directory as extract

path = extract.extract_dataset_dir()
detector = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml")
saving_path = "C:/Users/Student/Documents/training_data_pre_processed"

def PreProcessAndSaveImages(path):
    # Get pathes off all id directories
    id_pathes = os.listdir(path)
    for i in range(0,len(id_pathes),1):
        # Get pathes off all pictures in id directory
        tmp_pic_path_list = os.listdir(path + "/" + id_pathes[i])
        img_name = 1
        #Loop over each directory
        for tmp_pic_path in tmp_pic_path_list:
            # Grayscale images
            PIL_img = Image.open(path + "/" + id_pathes[i] + "/" + tmp_pic_path).convert('L') 
            # Convert into np.array
            img_numpy = np.array(PIL_img, 'uint8')
            # Detect faces in image
            faces = detector.detectMultiScale(img_numpy)
            #Draw rectangle around faces
            for(x,y,w,h) in faces:
                cv2.rectangle(img_numpy, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            # Transform back into image
            im = Image.fromarray(img_numpy[y:y+h,x:x+w])
            # Save into ids folder
            im.save(saving_path + "/" + id_pathes[i] + "/" + id_pathes[i] + "." + str(img_name) + ".jpeg")    
            img_name += 1
        # %-Status printed in terminal
        status = round(float(i/len(id_pathes)*100),2)
        print("Status pre-processing and saving: " + str(status) + "%" )

PreProcessAndSaveImages(path)
