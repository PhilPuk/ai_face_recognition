import cv2
from PIL import Image
import os
import numpy as np
import time
from skimage import io
import sys
sys.path.append('C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition')
from src.facial_recognition import face_recognition_image as rec
from util.utils import msgWithTime

def printMessageWithTime(message, start_time=0.0, code="INFO"):
            print(f"[{code}] {message} - Time: {time.time() - start_time}s")

class Model():
    def __init__(self, model_path = "",haarcascade_path="C:/Users/Student/Documents/repos/ai_face_recognition/ai_face_recognition/dataset/haarcascade_frontalface_default.xml", names_txt_path = ""):
        '''
        INIT of model class. Creates a cv2.CascadeClassifier and a cv2.face.LBPHFaceRecognizer_create
        @param model_path: Where the model should be saved, standart set to "".
        @param haarcascade_path: Path to file :"haarcascade_frontalface_default.xml". Downloadable in the internet, or included in cv2/data.
        @param Path to name id list.
        '''
        self.cascadePath =  haarcascade_path
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = model_path
        self.names = names_txt_path
        if model_path != "":
            try:
                self.recognizer.read(model_path)
            except:
                msgWithTime("Could not load model from given model_path.", 0)

    def loadModel(self, model_path):
        '''
        Load already existing model.
        @param model_path: Path of the model.
        '''
        self.model_path = model_path
        self.recognizer.read(model_path)
    
    def trainExistingModel(self, data_set_path, epochs, already_trained_epochs=0):
        '''
        Trains existing model.
        @param data_set_path: Path to the dataset of pre-processed Images. 
        Dataset should have the following directory order: .../dataset/"id"/"id"."numberOfPictureInOrder".extension.
        Example: .../dataset/230/230.1.jpg .
        @param epochs: For how many epochs should the model be trained.
        '''
        def printMessageWithTime(message, start_time=0.0):
            print(f"[INFO] {message} - Time of epoch: {time.time() - start_time}s")
        def getImagesAndLabelsPreProcessed(path):
            id_pathes = os.listdir(path)
            ids = []
            face_Samples = []
            id = 1
            for id_path in id_pathes:
                tmp_pic_path_list = os.listdir(path + "/" + id_path)
                for tmp_pic_path in tmp_pic_path_list:
                    PIL_img = Image.open(path + "/" + id_path + "/" + tmp_pic_path)
                    img_numpy = np.array(PIL_img, 'uint8')
                    face_Samples.append(img_numpy)
                    ids.append(id)
                id += 1
            return face_Samples, ids
        def trainer(model_path, recognizer, faces, ids):
            recognizer.update(faces,np.array(ids))
            recognizer.write(model_path)
        def calculateTimeForTraining(epochs,already_trained_epochs=0):
            '''
            Returns approximately the time it takes to train all epochs as exponential value and linear value.
            Only works if you know how many epochs were already trained!
            Formula lin: y = 0,9358*x+25,719
            @param: epochs: Amount of epochs.
            @param: already_trained_epochs: Amount of epochs that the model got trained before.
            '''
            total_lin = 0.0
            for i in range(already_trained_epochs,epochs+already_trained_epochs,1):
                total_lin += 0.9358*i+25.179
            return total_lin
        
        before_time = time.time()
        total_time = 0.0
        approx_total_time_lin = calculateTimeForTraining(epochs, already_trained_epochs)
        print(f"Approximately total time will be: {round(approx_total_time_lin,3)}s")
        faces, ids = getImagesAndLabelsPreProcessed(data_set_path)
        for i in range(epochs):
            start_time = time.time()
            trainer(self.model_path, self.recognizer, faces, ids)
            end_time = time.time()
            total_time += end_time - start_time
            try: 
                percent_lin = round(total_time/approx_total_time_lin*100,2)
            except:
                percent_lin = "Error!"
            printMessageWithTime(f"Epoch time: {i+1}/{epochs} - Status %: {percent_lin}", start_time=start_time)
        print(f"Training completed - Total training time: {time.time() - before_time} - Calculated time correctness: {round(total_time / approx_total_time_lin,2)*100}")
        
    def createModel(self, model_path, data_set_path):
        '''
        Creates a new model and saves it at the given location.
        @param model_path: Path where the model should be saved, including the name and excluding the file extension!
        @param data_set_path: Path to the dataset of pre-processed Images. 
        Dataset should have the following directory order: .../dataset/"id"/"id"."numberOfPictureInOrder".extension.
        Example: .../dataset/230/230.1.jpg .
        '''
        print("Creating model!")
        def getImagesAndLabelsPreProcessed(path):
            id_pathes = os.listdir(path)
            ids = []
            face_Samples = []
            id = 1
            for index, id_path in enumerate(id_pathes):
                tmp_pic_path_list = os.listdir(path + "/" + id_path)
                for tmp_pic_path in tmp_pic_path_list:
                    PIL_img = Image.open(path + "/" + id_path + "/" + tmp_pic_path)
                    img_numpy = np.array(PIL_img, 'uint8')
                    face_Samples.append(img_numpy)
                    ids.append(id)
                print(f"[LOADING DATASET] Status: {index+1} / {len(id_pathes)} - Status %: {round((index+1)/len(id_pathes)*100,2)}")
                id += 1
            return face_Samples, ids
        def trainer(model_path, recognizer, faces, ids):
            recognizer.train(faces,np.array(ids))
            recognizer.write(model_path)
        faces, ids = getImagesAndLabelsPreProcessed(data_set_path)
        trainer(model_path,self.recognizer,faces,ids)
        print("Model created!")
    
    def predict_and_show_image(self, img_path):
        '''
        Predicts who the person(s) on the image is(are) and show the image in a window.
        @param img_path: The path of the image to predict of.
        '''
        start_time = time.time()
        printMessageWithTime(f"Starting prediction for: {img_path}",start_time=start_time)
        img = io.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img, faces = rec.face_detecting(img, gray_img, minneig=8)
        # Display the output
        img = rec.face_recognition_image(img,gray_img, faces, self.recognizer, font_scale = 1.5)
        printMessageWithTime("Finished prediction",start_time=start_time)
        rec.printingWindow(img)
      
#def predictionEditedImages(path):
    # img_pathes = os.listdir(path)
    # for img_path in img_pathes:
    #     model.predict_and_show_image(path + img_path)
        
# Test sample
#model = Model()
#model.createModel("C:/Users/Student/Documents/models/personal_model.xml", "C:/Users/Student/Documents/datasets/personal_set/pre_processed")
#model.trainExistingModel("C:/Users/Student/Documents/datasets/pp_15_set", 500, 0)

#predictionEditedImages("C:/Users/Student/Documents/micro_data_set/images/2/")
#predictionEditedImages("C:/Users/Student/Desktop/edited_images/")