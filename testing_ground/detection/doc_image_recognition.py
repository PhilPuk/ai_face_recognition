import cv2
from skimage import io

def face_detection(img, gray_scaled_img,  putRectangle, minneig=5 , haardcascade_path="./dataset/haarcascade_frontalface_default.xml"):
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
    if putRectangle:
        # Draw rectangle around the faces  
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    
    return img, faces

def face_recognition_image(img, gray_scaled_image, recognizer_model, person_dict, font=cv2.FONT_HERSHEY_TRIPLEX, font_scale=1):
    '''
    Returns the img with a name, id and the confidence of the predicion on it.
    @param img: img returned from face_detecting_images.
    @param gray_scaled_image: normal image just gray scaled.
    @param recognizer_model: Created model of your recognizer.
    @param person_dict: Dictionary of persons you get from get_person_dict.py
    @param font: cv2 Font.
    '''
    img, faces = face_detection(img, gray_scaled_image, False)
    
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
    cv2.destroyAllWindows()
    
def getNamesAndIDSfromTXT(names_txt_path):
    with open(names_txt_path, "r") as f:
        lines = f.readlines()
        persons = dict()
        for line in lines:
            info = line.split()
            persons[info[2]] = f"{info[0]} {info[1]}"
        return persons

# Load up our model change the path to your models path
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./models/my_model.xml")
# initialize our person dict from the names.txt file of our dataset    
person_dict = getNamesAndIDSfromTXT("./my_dataset/names.txt")
# Load up image from given path
image = io.imread("./images/phil_p_3.jpg")
# Grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Do prediction on our image
recognition_image = face_recognition_image(image, gray_image, recognizer, person_dict)
# Create window to show our image with the prediciton
printingWindow(recognition_image)


