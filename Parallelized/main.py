import sys
import traceback
from skimage import io
import cv2
sys.path.append('./')
import threading
from multiprocessing import Pool
import Parallelized.misc as misc
import classes.camera as Cam
from classes.model import Model


class System:
    def __init__(self):
        self.haardcascade_path = "./dataset/haarcascade_frontalface_default.xml"
        self.cv2.CascadeClassifier(self.haardcascade_path)
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.model = Model

    def main(self):
        cam = Cam(640,360)
        cam.cam = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, img = cam.cam.read()
            if not ret:
                print("Empty frame")
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Draw a rectangle around the faces
                img, faces = 0,0
                with Pool() as pool:
                    if not pool._check_running():
                        pool.map_async(misc.face_detection,1)
                        pool.apply(img, gray, )
                with Pool() as pool:
                    if not pool._check_running():
                    pool.map_async(misc.face_recognition_image, img, faces,1)

                # Display resulting frame
                cv2.imshow("camera", img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        # Free memory and close programm
        cam.cam.release()
        cv2.destroyAllWindows()

system = System()

system.main()