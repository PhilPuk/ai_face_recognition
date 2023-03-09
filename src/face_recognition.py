import cv2 
from skimage import io 

print(cv2.__version__)

#Returns passed image with rectangles around faces
#Parameter: Path: Path of picture
def face_recognition_image(path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #Load image
    img = io.imread(path)

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
    #cv2.imshow('image', img)  
    #cv2.waitKey()  
    return img