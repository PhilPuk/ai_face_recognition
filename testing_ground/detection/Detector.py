import numpy as np
import cv2

class Detector:
    def __init(self):
        self.ModelProtoPath = "C:/Users/Student/Documents/models/res10_300x300_ssd_iter_140000.prototxt"
        self.faceModel = cv2.dnn.readNetFromCaffe(self.ModelProtoPath)
        caffeModel = "C:/Users/Student/Documents/models/res10_300x300_ssd_iter_140000.caffemodel"
        #read model
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]
        
        self.processFrame()
        cv2.imshow("Output", self.img)
        cv2.waitkey(0)
        
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