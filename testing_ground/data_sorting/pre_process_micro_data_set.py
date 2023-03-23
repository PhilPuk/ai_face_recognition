import cv2
import numpy as np
from PIL import Image
import os
import sys

def PreProcessAndSaveImages(src_path, saving_path):
    detector = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml")
    # Get pathes off all id directories
    dir_list = os.listdir(src_path) 
    for i in dir_list:
        img_name = 1
        #Loop over each directory
        for tmp_pic_path in os.listdir(src_path + i):
            # Grayscale images
            PIL_img = Image.open(src_path + i + "/" + tmp_pic_path).convert('L') 
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
            im.save(saving_path + i + "/" + i + "." + str(img_name) + ".jpeg")    
            img_name += 1
        # %-Status printed in terminal
        #status = round(float(i/len(dir_list)*100),2)
        #print("Status pre-processing and saving: " + str(status) + "%" )
        
# Test sample
PreProcessAndSaveImages("C:/Users/Student/Documents/micro_data_set/images/", "C:/Users/Student/Documents/pre_processed_micro_data_set/")