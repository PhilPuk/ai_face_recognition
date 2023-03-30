import cv2
import sys
sys.path.append('./')
from util.utils import msgWithTime


def main(cam):
    msgWithTime("Initializing camera.", 1)
    #Init cam
    cam.cam = cv2.VideoCapture(0)
    msgWithTime("Camera initialized.", 1)

    while True:
        ret, img = cam.cam.read()
        if ret:
            cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.cam.release()
    cv2.destroyAllWindows()
    msgWithTime("Exiting program", 1)
