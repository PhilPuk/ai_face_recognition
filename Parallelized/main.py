import sys
import traceback
from skimage import io
import cv2
sys.path.append('./')
import Parallelized.misc as misc
import 


class System:
    def __init__(self):
        self.haardcascade_path = "./dataset/haarcascade_frontalface_default.xml"
        self.cv2.CascadeClassifier(self.haardcascade_path)
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        
    def main(self):
        cam.cam = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, img = cam.cam.read()
            if not ret:
                print("Empty frame")
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Draw a rectangle around the faces
                # Display resulting frame
                cv2.imshow("camera", img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        # Free memory and close programm
        cam.cam.release()
        cv2.destroyAllWindows()