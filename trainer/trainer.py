import cv2
import numpy as np
from PIL import Image
import os
import sys

sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.extract_dataset_directory as extract
import src.image_ids as dataCollector
path = extract.extract_dataset_dir("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml")
#recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml")

images, ids = dataCollector.getIDS("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt")

for i in images:
    print(i)