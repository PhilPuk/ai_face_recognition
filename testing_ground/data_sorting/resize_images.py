import os
import cv2
import numpy as np
from PIL import Image

def resizeImagesInDir(src_path,new_path, resolution):
    for file in os.listdir(src_path):
        try:
            image = Image.open(src_path + file)
            new_image = image.resize((resolution[0],resolution[1]))
            new_image.save(new_path + file)
        except:
            print("Could not open Image at: " + src_path + file)
    print("Resizing finished!")

# Test sample
resizeImagesInDir("C:/Users/Student/Desktop/images/","C:/Users/Student/Desktop/new/", (179,307))