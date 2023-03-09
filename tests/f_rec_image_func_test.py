import sys
import cv2
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
import src.face_recognition as fr

img = fr.face_recognition_image("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/test2.jpg")

cv2.imshow('image', img)  
cv2.waitKey()