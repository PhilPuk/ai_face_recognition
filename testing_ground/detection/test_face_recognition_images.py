import cv2 
from skimage import io 

print(cv2.__version__)

cascadePath = "C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascadePath)
  
# Read the input image    
#img = io.imread('PhilPuk/ai_face_recognition/testing/test.jpg')
# FULLPATH:'PhilPuk/ai_face_recognition/testing/test.jpg'
# NAME PATH:"test.jpg"
img = io.imread("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/test2.jpg")

# Convert into grayscale  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

# Detect faces  
faces = face_cascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        ) 
  
# Draw rectangle around the faces  
for (x, y, w, h) in faces:  
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
  
# Display the output  
cv2.imshow('image', img)  
cv2.waitKey()  