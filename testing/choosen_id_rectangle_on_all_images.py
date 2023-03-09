import cv2
import os
from skimage import io
import sys
import cv2
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.face_recognition as fr

id = "3214"
dir_images = "C:/Users/Student/Documents/img_align_celeba/"
dir_txt = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/identity_CelebA.txt"

img_list = []
img_faces_with_rectangles = []

with open(dir_txt) as f:
        # Collect all lines of the txt
        lines = f.readlines()
        for line in lines:
            text = line.split()
            if text[1] == id:
                img_list.append(text[0])

print(img_list)

for i in img_list:
    img_faces_with_rectangles.append(fr.face_recognition_image(dir_images + i))

for i in range(len(img_list)):
    cv2.imshow(f'image{i}', img_faces_with_rectangles[i])  
cv2.waitKey()  