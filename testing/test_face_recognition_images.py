import cv2 
from skimage import io 

print(cv2.__version__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
# Read the input image    
#img = io.imread('PhilPuk/ai_face_recognition/testing/test.jpg')
# FULLPATH:'PhilPuk/ai_face_recognition/testing/test.jpg'
# NAME PATH:"test.jpg"
img = io.imread("C:/Users/Student/Documents/AI_Face_Recognition/ai_face_recognition/ai_face_recognition/testing/test2.jpg")

# Convert into grayscale  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

# Detect faces  
faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
  
# Draw rectangle around the faces  
for (x, y, w, h) in faces:  
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
  
# Display the output  
cv2.imshow('image', img)  
cv2.waitKey()  