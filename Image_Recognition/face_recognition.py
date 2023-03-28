import cv2
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from Detection import face_detection

def face_recognition_image(img, gray_scaled_image, recognizer_model, person_dict, font=cv2.FONT_HERSHEY_TRIPLEX):
    '''
    Returns the img with a name, id and the confidence of the predicion on it.
    @param img: img returned from face_detecting_images.
    @param gray_scaled_image: normal image just gray scaled.
    @param recognizer_model: Created model of your recognizer.
    @param font: cv2 Font.
    '''
    img, faces = face_detection.face_detection(img, gray_scaled_image)
    
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
        cv2.putText(img, str(id), (x+5,y-5),font,2,(255,255,255),2)
        cv2.putText(img,str(confidence),(x+5,y+h-5),font,2,(255,255,0),1)
    return img