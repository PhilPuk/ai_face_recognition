import cv2
# ADD PERSON DICT

def getNamesAndIDSfromTXT(names_txt_path):
    '''
    Returns a dictionary of our ids and the perons that correspond to them.
    @param names_txt_path: Path to our names.txt file of our dataset.
    '''
    # Opens txt file in reading mode
    with open(names_txt_path, "r") as f:
        # Read all lines of txt in a list
        lines = f.readlines()
        # Initialize dictionary
        persons = dict()
        # Loop over each line of txt
        for line in lines:
            # Seperate line info
            info = line.split()
            # Add information to dictionary 
            persons[info[2]] = f"{info[0]} {info[1]}"
        # Return our dictionary
        return persons

class Camera():
    def __init__(self, resolution):
        self.width = resolution[0]
        self.height = resolution[1]
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3,self.width)
        self.cam.set(4,self.height)
        self.minW = 0.25*self.cam.get(3)
        self.minH = 0.25*self.cam.get(4)
# Initializing our camera 
Cam = Camera([640,480])
# Loading up the haarcascade_frontalface_default.xml file
faceCascade = cv2.CascadeClassifier("./dataset/haarcascade_frontalface_default.xml")
# Load up our model change the path to your models path
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./models/my_model.xml")
font = cv2.FONT_HERSHEY_TRIPLEX
# Load our person dictionary
person_dict = getNamesAndIDSfromTXT("./my_dataset/names.txt")
#Init cam with resolution 640x480

# Main loop
while True:
    # Read camera frame by frame
    ret, img = Cam.cam.read()
    # Check if camera frame exists
    if ret:
        # Convert frame to grayscaled frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(int(Cam.minW),int(Cam.minH)),
            maxSize=(400,400)
        )
        # Initializing variables for storing person id and confidence of prediction
        id = 0
        confidence = 0
        # Loop over detected faces
        for(x,y,w,h) in faces:
            # Put rectangles on faces
            cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
            # Predict person inside rectangles
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if the person is known
            if confidence < 100:
                if id <= len(person_dict):
                    id = person_dict[str(id)] + " - " + str(id)
                else:
                    id = str(id)
            else:
                id = "unknown"
            # Create text for confidence
            confidence = "  {0}%".format(round(100-confidence))
            # Put confidence and person name + id on the frame
            cv2.putText(img, str(id), (x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)
        # Display frame to camera window
        cv2.imshow('camera',img)
    k = cv2.waitKey(20)
    if k == 27:
        break
Cam.release()
cv2.destroyAllWindows()
