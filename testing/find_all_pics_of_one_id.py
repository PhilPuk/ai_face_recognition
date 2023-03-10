import cv2
import os
from skimage import io
import sys
import cv2
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.face_recognition as fr
import src.extract_dataset_directory as extractor

#Returns list off all image file names found.
#Takes an int for the id to search.
#Takes bool to show images or not.
def findAllPicturesOf_ID(id, showImages):
    id = "1"
    dir_images = extractor.extract_dataset_dir("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/data_set_dir.xml")
    dir_txt = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt"

    img_list = []
    loaded_images = []

    with open(dir_txt) as f:
        # Collect all lines of the txt
        lines = f.readlines()
        for line in lines:
            text = line.split()
        if text[1] == id:
            img_list.append(text[0])

    print(img_list)
    if showImages:
        for i in img_list:
            loaded_images.append(io.imread(dir_images + i))

        for i in range(len(img_list)):
            cv2.imshow(f'image{i}', loaded_images[i])  
        cv2.waitKey()
    return img_list