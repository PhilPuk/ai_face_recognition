import sys
import cv2
sys.path.append('./')
from Detection import face_detection as det
from Image_Recognition import face_recognition as rec

def main(model, cam, person_dict):
    cam.cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.cam.read()
        if ret:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = rec.face_recognition_image(img, gray_img, model.recognizer, person_dict)
            cv2.imshow("app", img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cam.cam.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Exiting program")