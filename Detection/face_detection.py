import cv2 

def face_detection(img, gray_scaled_img,  minneig=5 , haardcascade_path="C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml"):
    '''
    Returns passed image, faces tuple. Img with rectangles around faces, aswell as the vectors and coordinates for the detected faces such as: x,y,width,height.
    @param img: loaded img, with PIL, skimage or cv2, @param minneig: Higher numbers reduces chances of false detection.
    @param haarcascade_path: Path to file :"haarcascade_frontalface_default.xml". Downloadable in the internet, or included in cv2/data.
    '''
    face_cascade = cv2.CascadeClassifier(haardcascade_path)

    # Detect faces  
    faces = face_cascade.detectMultiScale(
            gray_scaled_img,     
            scaleFactor=1.2,
            minNeighbors=minneig,     
            minSize=(20, 20)
        ) 
  
    # Draw rectangle around the faces  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
  
    return img, faces