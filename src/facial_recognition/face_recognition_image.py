import cv2 
from skimage import io 

def face_detecting(img, gray_scaled_img,  minneig=5):
    '''
    Returns passed image with rectangles around faces, aswell as the vectors and coordinates for the detected faces such as: x,y,width,height.
    @param img: loaded img, with PIL, skimage or cv2, @param minneig: Higher numbers reduces chances of false detection.
    '''
    face_cascade = cv2.CascadeClassifier("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml")

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

def printingWindow(img, window_title="Window", resizable=True):
    '''
    Creates a window where the given image will be shown.
    @param img: img to show. @param window_title: Name of the window. @param resizable: Should the window be able to change its size.
    '''
    if resizable:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(window_title)
        
    cv2.imshow(window_title,img)
    cv2.waitKey()
    
    
def face_recognition_image(img, gray_scaled_image, faces, recognizer_model, font=cv2.FONT_HERSHEY_TRIPLEX, font_scale=1):
    '''
    Returns the img with a name, id and the confidence of the predicion on it.
    @param img: img returned from face_detecting_images. @param gray_scaled_image: normal image just gray scaled. @param faces: detected faces as tuple such as: x,y,width,height. 
    @param recognizer_model: Created model of your recognizer. @param font: cv2 Font.
    '''
    names = ["None", "John", "Phil", "Stew", "Stacy", "Lee"]
    for(x,y,w,h) in faces:
        try:
            id, confidence = recognizer_model.predict(gray_scaled_image[y:y+h,x:x+w])
            if confidence < 100:
                if id <= len(names):
                    id = names[id] + " - " + str(id)
                else:
                    id = str(id)
            else:
                id = "unknown"
            confidence = "  {0}%".format(round(100-confidence))
            cv2.putText(img, str(id), (x+5,y-5),font,font_scale,(255,255,255),2)
            cv2.putText(img,str(confidence),(x+5,y+h-5),font,font_scale,(255,255,0),2)
        except:
            print("[ERROR] Could not predict, due to unkown error!")
    return img