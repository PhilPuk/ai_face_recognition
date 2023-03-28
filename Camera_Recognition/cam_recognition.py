import sys
import cv2
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from classes.model import Model
from classes.camera import Camera
from util.utils import msgWithTime
from Detection import face_detection as det
from Image_Recognition import face_recognition as rec
from src.misc.get_person_dict import getNamesAndIDSfromTXT

def main():
    msgWithTime("Loading person dict.", "INFO")
    persons_dict = getNamesAndIDSfromTXT("C:/Users/Student/Documents/datasets/personal_set/names.txt")
    msgWithTime("Loaded person dict - Initializing camera.", "INFO")
    cam = Camera([640,480])
    msgWithTime("Camera loaded.", "INFO")
    model = Model("C:/Users/Student/Documents/models/personal_model.xml")
    while True:
        ret, img = cam.cam.read()
        if ret:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = rec.face_recognition_image(img, gray_img, model.recognizer, persons_dict)
            cv2.imshow("app", img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cam.cam.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Exiting program")
    
main()
        
    
    
    