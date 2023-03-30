import cv2
import numpy as np
from PIL import Image
import os

# Variable to set our model name, dont change .xml
model_name = f"my_model.xml"
# Variable of our pre_processed folder from our dataset
path_pre_processed_images = "./my_dataset/pre_processed"
# Creating the recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Loading up the haarcascade_frontalface_default.xml file
detector = cv2.CascadeClassifier("./dataset/haarcascade_frontalface_default.xml")

def getImagesAndLabelsPreProcessed(path):
    '''
    Returns the faceSamples and ids of our pre-processed images.
    @param Path: path of our pre-processed images from the dataset
    '''
    # List id folders at given paths
    id_pathes = os.listdir(path)
    # Initialize empty lists for data
    ids = []
    face_Samples = []
    # ID counter
    id = 1
    # Loop over each id folder
    for id_path in id_pathes:
        # List images names
        tmp_pic_path_list = os.listdir(path + "/" + id_path)
        # Loop over each image
        for tmp_pic_path in tmp_pic_path_list:
            # Load image and add to our faceSamples list aswell as adding the id to our id list.
            PIL_img = Image.open(path + "/" + id_path + "/" + tmp_pic_path)
            img_numpy = np.array(PIL_img, 'uint8')
            face_Samples.append(img_numpy)
            ids.append(id)
        id += 1
    return face_Samples, ids
# Call our function to get the data
faces ,ids = getImagesAndLabelsPreProcessed(path_pre_processed_images)
# Train model
recognizer.train(faces,np.array(ids))
# Save model
recognizer.write(f"{model_name}")


