import numpy as np
import cv2
from imutils.video import FPS


class Detector:
    def __init__(self, use_cuda = False):
        self.faceModel = cv2.dnn.readNetFromCaffe("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/detection/res10_300x300_ssd_iter_140000.prototxt.txt",
        caffeModel="C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/detection/res10_300x300_ssd_iter_140000.caffemodel")
        
        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]
        
        self.processFrame()
        cv2.imshow("Output", self.img)
        cv2.waitKey(0)
    
    def processWebCam(self):
        appname = "Face_Detection"
        cv2.namedWindow(appname)
        vc = cv2.VideoCapture(0)
        vc.set(3,640)
        vc.set(4,480)
        fps = FPS().start()
        while True:
            ret, self.img = vc.read()
            # If frame exists
            if ret:
                (self.height, self.width) = self.img.shape[:2]
                self.processFrame()
                cv2.imshow(appname, self.img)
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
                fps.update()
        fps.stop()
        print(f"Elapsed time: {round(fps.elapsed(),2)}")
        print("FPS: {:.2f}".format(fps.fps()))
        # Free memory and close programm
        vc.release()
        cv2.destroyWindow(appname)
        
    def processFrame(self):
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300,300), (104.0, 177.0, 123.0),
        swapRB = False, crop = False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        
        for i in range(0, predictions.shape[2]):
            if predictions[0,0,i,2] > 0.5:
                bbox = predictions[0,0,i,3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")
                cv2.rectangle(self.img, (xmin,ymin), (xmax,ymax), (0,0,255),2)

detector = Detector()
# Image detector test
#detector.processImage("C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/testing/detection/test.jpg")
# WebCam detector test
detector.processWebCam()
