import sys
import traceback
from skimage import io
import cv2
sys.path.append('./')

def face_detection(img, gray_scaled_img,  putRectangle, face_cascade, minneig=5):
    '''
    Returns passed image, faces tuple. Img with rectangles around faces, aswell as the vectors and coordinates for the detected faces such as: x,y,width,height.
    @param img: loaded img, with PIL, skimage or cv2, @param minneig: Higher numbers reduces chances of false detection.
    @param face_cascade: loaded object: cv2.CascadeClassifier.
    @param minneig: Higher number = less false detection.
    '''
    # Detect faces  
    faces = face_cascade.detectMultiScale(
            gray_scaled_img,     
            scaleFactor=1.2,
            minNeighbors=minneig,     
            minSize=(20, 20)
        ) 
    if putRectangle:
        # Draw rectangle around the faces  
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    
    return img, faces

def face_recognition_image(img, faces, gray_scaled_image, recognizer_model, person_dict, font, font_scale=1):
    '''
    Returns the img with a name, id and the confidence of the predicion on it.
    @param img: img returned from face_detecting_images.
    @param faces: 2nd return value of face_detection().
    @param gray_scaled_image: normal image just gray scaled.
    @param recognizer_model: Created model of your recognizer.
    @param person_dict: Dictionary of persons you get from get_person_dict.py
    @param font: cv2 Font.
    '''
    
    for(x,y,w,h) in faces:
        id, confidence = recognizer_model.predict(gray_scaled_image[y:y+h,x:x+w])
        if confidence < 100:
            if id <= len(person_dict):
                id = person_dict[str(id)] + " - " + str(id)
            else:
                id = str(id)
        else:
            id = "unknown"
        confidence = "  {0}%".format(round(100-confidence))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        cv2.putText(img, str(id), (x+5,y-5),font,font_scale,(255,255,255),2)
        cv2.putText(img,str(confidence),(x+5,y+h-5),font,font_scale,(255,255,0),1)
    return img