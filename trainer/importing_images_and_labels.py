import numpy as np
from PIL import Image
import os

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    pictures = []
    person_id = []
    picture_id = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # load image and grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id 